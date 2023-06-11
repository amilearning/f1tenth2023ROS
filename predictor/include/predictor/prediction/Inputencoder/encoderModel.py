import torch
from torch import nn 
from torch.nn import functional as F

with torch.no_grad() and torch.cuda.amp.autocast():
        
    class AutoEncoder(nn.Module):
        def __init__(self,input_size = 8, hidden_size = 5, num_layers = 2):
            super(AutoEncoder, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=False,
            )
            
        def forward(self, x):        
            outputs, (hidden, cell) = self.lstm(x)
            return (hidden, cell)

    class AutoDecoder(nn.Module):
        def __init__(
            self, input_size=8, hidden_size=5, output_size=8, num_layers=2):
            super(AutoDecoder, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=False,
            )
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden = None):
            if hidden is None:
            # x: tensor of shape (batch_size, seq_length, hidden_size)
                output, (hidden, cell) = self.lstm(x)
            else:
                output, (hidden, cell) = self.lstm(x, hidden)
            prediction = self.fc(output)
            return prediction, (hidden, cell)
        # def forward(self, x):
        #     # x: tensor of shape (batch_size, seq_length, hidden_size)
        #     output, (hidden, cell) = self.lstm(x)

        #     return output, (hidden, cell)
            

    class LSTMAutomodel(nn.Module):
        """LSTM-based Auto Encoder"""

        def __init__(
            self, args):
            """
            args['input_size']: int, batch_size x sequence_length x input_dim
            args['hidden_size']: int, output size of LSTM VAE
            args['latent_size']: int, latent z-layer size
            num_lstm_layer: int, number of layers in LSTM
            """
            super(LSTMAutomodel, self).__init__()
            self.device = args['device']

            # dimensions
            self.input_size = args['input_size']
            self.hidden_size = args['hidden_size']
            self.latent_size = args['latent_size']
            self.seq_len = args['seq_len']
            self.num_layers = 1

            # lstm ae
            self.lstm_enc = AutoEncoder(
                input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers
            )
            self.lstm_dec = AutoDecoder(
                input_size=self.latent_size,
                output_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )

            self.fc21 = nn.Linear(self.hidden_size,self.hidden_size)                        
            self.relu = nn.ReLU()
            self.fc22 = nn.Linear(self.hidden_size, self.latent_size)                        
            
            

        def reparametize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std).to(self.device)

            z = mu + noise * std
            return z

        def get_latent_z(self,x):
            if len(x.shape) < 3:
                # evaluation -> batch is 1 
                batch_size = 1 
                seq_len, feature_dim = x.shape
            else:
                batch_size, seq_len, feature_dim = x.shape

            # encode input space to hidden space
            enc_hidden = self.lstm_enc(x)
            enc_h = enc_hidden[0].view(batch_size, self.hidden_size).to(self.device)
            # extract latent variable z(hidden space to latent space)
            z = self.fc22(self.relu(self.fc21(enc_h)))
            
            return z

        def forward(self, x):
        
            batch_size, seq_len, feature_dim = x.shape                
            if seq_len != self.seq_len:
                print("Warning !! sequence lentgh is not matched")
                return
                
            enc_hidden = self.lstm_enc(x)
            enc_h = enc_hidden[0].view(batch_size, self.hidden_size).to(self.device)
            # extract latent variable z(hidden space to latent space)
            z = self.fc22(self.relu(self.fc21(enc_h)))              
            # z = self.reparametize(mean, logvar)  # batch_size x latent_size
            # decode latent space to input space
            z = z.repeat(1, seq_len, 1)
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
            # reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)
            reconstruct_output, hidden = self.lstm_dec(z)
            # reconstruct_output, hidden = self.lstm_dec(z)
            x_hat = reconstruct_output
            # calculate vae loss
            losses = self.loss_function(x_hat, x)
            m_loss = losses["loss"]
            
            return m_loss, x_hat
                
        def loss_function(self, *args, **kwargs) -> dict:
            """
            Computes the VAE loss function.
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            :param args:
            :param kwargs:
            :return:
            """
            recons = args[0]
            input = args[1]
            recons_loss = F.mse_loss(recons, input)
            loss = recons_loss 
            return {
                "loss": loss
            }

        def loss_function_consensus(self, *args, **kwargs) -> dict:
            """
            Computes the VAE loss function.
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            :param args:
            :param kwargs:
            :return:
            """
            recons_loss = 0
            for i in range(len(args[2])-1):
                recons = args[0][i]
                input = args[1][i]
                recons_loss += F.mse_loss(recons, input)

            mu_consensus_loss = 0
            for i in range(len(args[2])-1):
                mu_consensus_loss += (args[2][i]-args[2][i+1])**2
            mu_consensus_loss_weight = 0.1

            loss = recons_loss +  mu_consensus_loss_weight*mu_consensus_loss
            return {
                "loss": loss,                
                "MeanConsensus_Loss": mu_consensus_loss.detach()
            }

            # recons = args[0][0]
            # input = args[1][0]            
            # recons_loss = F.mse_loss(recons, input)
            # loss = recons_loss
            # return {
            #     "loss": loss,                
            #     "MeanConsensus_Loss": torch.zeros(10)
            # }



# def Autotrain(args, model, train_loader, test_loader, writer):
#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

#     ## interation setup
#     epochs = tqdm(range(args['max_iter'] // len(train_loader) + 1))

#     ## training
#     count = 0
#     for epoch in epochs:
#         model.train()
#         optimizer.zero_grad()
#         train_iterator = tqdm(
#             enumerate(train_loader), total=len(train_loader), desc="training"
#         )

#         for i, batch_data in train_iterator:

#             if count > args['max_iter']:
#                 return model
#             count += 1

#             train_data = batch_data[:,:,:,0:-2].to(args['device'])

#             ## reshape
#             batch_size = train_data.size(0)
#             # example_size = past_data.size(1)
#             # image_size = past_data.size(1), past_data.size(2)
#             # past_data = (
#             #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
#             # )
#             # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
#             # m_loss, x_hat, mean_consensus_loss
#             mloss, recon_x, mean_consensus_loss = model(train_data)

#             # Backward and optimize
#             optimizer.zero_grad()
#             mloss.mean().backward()
#             optimizer.step()

#             train_iterator.set_postfix({"train_loss": float(mloss.mean())})
#             train_iterator.set_postfix({"consensus_loss": float(mean_consensus_loss.mean())})
#         writer.add_scalar("train_loss", float(mloss.mean()), epoch)
#         writer.add_scalar("consensus_loss", float(mean_consensus_loss.mean()), epoch)

#         model.eval()
#         eval_loss = 0
#         consensus_loss = 0
#         test_iterator = tqdm(
#             enumerate(test_loader), total=len(test_loader), desc="testing"
#         )

#         with torch.no_grad():
#             for i, batch_data in test_iterator:
#                 test_data = batch_data[:,:,:,0:-2].to(args['device'])

#                 ## reshape
#                 batch_size = test_data.size(0)
#                 # example_size = past_data.size(1)
#                 # past_data = (
#                 #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
#                 # )
#                 # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

#                 mloss, recon_x, mean_consensus_loss  = model(test_data)

#                 eval_loss += mloss.mean().item()
#                 consensus_loss +=mean_consensus_loss.mean().item()

#                 test_iterator.set_postfix({"eval_loss": float(mloss.mean())})
#                 test_iterator.set_postfix({"consensus_loss": float(mean_consensus_loss.mean())})

#                 # if i == 0:
#                 #     for idx in range(len(recon_x)):
#                 #         plt.plot(recon_x[idx,:,0].cpu(), recon_x[idx,:,0].cpu(), 'r.')
#                 #         plt.plot(past_data[idx,:,0].cpu(), past_data[idx,:,0].cpu(), 'g.')                        
#                 #         plt.pause(0.01)
#                 #     plt.clf()
#                     # nhw_orig = past_data[0].view(example_size, image_size[0], -1)
#                     # nhw_recon = recon_x[0].view(example_size, image_size[0], -1)                    
#                     # writer.add_images(f"original{i}", nchw_orig, epoch)
#                     # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)

#         eval_loss = eval_loss / len(test_loader)
#         writer.add_scalar("eval_loss", float(eval_loss), epoch)
#         writer.add_scalar("consensus_loss", float(consensus_loss), epoch)        
#         print("Evaluation Score : [{}]".format(eval_loss))
#         print("consensus_loss : [{}]".format(consensus_loss))

#     return model
