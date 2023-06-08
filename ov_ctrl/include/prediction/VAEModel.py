import torch
from torch import nn 
from torch.nn import functional as F

with torch.no_grad() and torch.cuda.amp.autocast():
        
    class VAEEncoder(nn.Module):
        def __init__(self,input_size = 8, hidden_size = 5, num_layers = 2):
            super(VAEEncoder, self).__init__()
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

    class VAEDecoder(nn.Module):
        def __init__(
            self, input_size=8, hidden_size=5, output_size=8, num_layers=2):
            super(VAEDecoder, self).__init__()
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
        
        def forward(self, x, hidden):
            # x: tensor of shape (batch_size, seq_length, hidden_size)
            output, (hidden, cell) = self.lstm(x, hidden)
            prediction = self.fc(output)
            return prediction, (hidden, cell)
            

    class LSTMVAE(nn.Module):
        """LSTM-based Variational Auto Encoder"""

        def __init__(
            self, args):
            """
            args['input_size']: int, batch_size x sequence_length x input_dim
            args['hidden_size']: int, output size of LSTM VAE
            args['latent_size']: int, latent z-layer size
            num_lstm_layer: int, number of layers in LSTM
            """
            super(LSTMVAE, self).__init__()
            self.device = args['device']

            # dimensions
            self.input_size = args['input_size']
            self.hidden_size = args['hidden_size']
            self.latent_size = args['latent_size']
            self.num_layers = 1

            # lstm ae
            self.lstm_enc = VAEEncoder(
                input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers
            )
            self.lstm_dec = VAEDecoder(
                input_size=self.latent_size,
                output_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )

            self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
            self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
            self.fc3 = nn.Linear(self.latent_size, self.hidden_size)

        def reparametize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std).to(self.device)

            z = mu + noise * std
            return z

        def get_theta_dist(self,x):
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
            mean = self.fc21(enc_h)
            logvar = self.fc22(enc_h)

            return mean, logvar 

        def forward(self, x):
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
            mean = self.fc21(enc_h)
            logvar = self.fc22(enc_h)
            z = self.reparametize(mean, logvar)  # batch_size x latent_size

            # decode latent space to input space
            z = z.repeat(1, seq_len, 1)
            z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
            reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)

            x_hat = reconstruct_output

            # calculate vae loss
            losses = self.loss_function(x_hat, x, mean, logvar)
            m_loss, recon_loss, kld_loss = (
                losses["loss"],
                losses["Reconstruction_Loss"],
                losses["KLD"],
            )

            return m_loss, x_hat, (recon_loss, kld_loss), mean, logvar


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
            mu = args[2]
            log_var = args[3]

            kld_weight = 0.00025  # Account for the minibatch samples from the dataset
            recons_loss = F.mse_loss(recons, input)

            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )

            loss = recons_loss + kld_weight * kld_loss
            return {
                "loss": loss,
                "Reconstruction_Loss": recons_loss.detach(),
                "KLD": -kld_loss.detach(),
            }

