import torch
from torch import nn 
from torch.nn import functional as F


    
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
        

class ContLSTMAutomodel(nn.Module):
    """LSTM-based Contrasiave Auto Encoder"""

    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(ContLSTMAutomodel, self).__init__()
        self.device = args['device']

        # dimensions
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.latent_size = args['latent_size']
        self.seq_len = args['seq_len']
        self.num_layers = 2

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

        self.fc_l2l = nn.Linear(self.latent_size, self.latent_size*self.seq_len)

        self.fc21 = nn.Linear(self.hidden_size,self.hidden_size)                        

        self.relu = nn.ReLU()
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)                        
        
        



    def get_latent_z(self,x):
        if len(x.shape) < 3:
            # evaluation -> batch is 1 
            batch_size = 1 
            seq_len, feature_dim = x.shape
            x = x.unsqueeze(dim=0)
        else:
            batch_size, seq_len, feature_dim = x.shape

        # encode input space to hidden space
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0][-1,:,:].view(batch_size, self.hidden_size).to(self.device)
        # extract latent variable z(hidden space to latent space)
        z = self.fc22(self.relu(self.fc21(enc_h)))
        
        return z

    def forward(self, x):
    
        batch_size, seq_len, feature_dim = x.shape                
        if seq_len != self.seq_len:
            print("Warning !! sequence lentgh is not matched")
            return
            
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0][-1,:,:].view(batch_size, self.hidden_size).to(self.device)
        # extract latent variable z(hidden space to latent space)
        z = self.fc22(self.relu(self.fc21(enc_h)))       
        ### 
        theta = z.clone()
        z = self.fc_l2l(z.to(self.device))        
        # decode latent space to input space
        # z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
        # reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)
        reconstruct_output, hidden = self.lstm_dec(z)
        # reconstruct_output, hidden = self.lstm_dec(z)
        x_hat = reconstruct_output
        
        # calculate  loss
        losses = self.loss_cont(x_hat, x, theta)
        m_loss = losses["loss"]
        cont_loss = losses["cont_loss"]
        recon_loss = losses["recons_loss"]
        
        return m_loss, x_hat, cont_loss, recon_loss

    def compute_euclidian_dist(self,A,B):
        # A = torch.randn(512, 5, 9)
        # B = torch.randn(512, 5, 9)
        # Expand dimensions to enable broadcasting
        A_expanded = A.unsqueeze(1)  # [512, 1, 4, 7]
        B_expanded = B.unsqueeze(0)  # [1, 512, 4, 7]
        # Calculate the Euclidean norm between each pair of vectors
        distances = torch.norm((A_expanded - B_expanded), dim=-1)  # [512, 512, 4]        
        # Sum the Euclidean norms over the sequence dimension
        if len(A.shape) > 2:            
            seq_sum_distances = torch.sum(distances, dim=2)  # [512, 512]            
        else:
            seq_sum_distances = distances  # [512, 512]
        normalized_tensor = F.normalize(seq_sum_distances, dim=(0, 1))
        return normalized_tensor      

    def loss_cont(self,*args, **kwargs) -> dict:
        """
        Computes loss
        loss = contrasive_loss + reconstruction loss 
        """
        recons = args[0]
        input = args[1]
        theta = args[2]
        diff_input = input[:,1:,[0,1,2,3,4,5,6]] - input[:,0:-1,[0,1,2,3,4,5,6]]
        input_diff_mean = self.compute_euclidian_dist(diff_input,diff_input) # input_diff_mean = batch x batch 
        theta_diff_mean = self.compute_euclidian_dist(theta,theta) # theta_diff_mean = batch x batch 
        cont_loss = F.mse_loss(torch.exp(input_diff_mean),torch.exp(theta_diff_mean))            
        # cont_loss_scale_factor_weighted = cont_loss * 10**int(torch.log10((recons_loss/cont_loss)))
        cont_loss_weight_multiplier = 5e2
        cont_loss_weighted = cont_loss*cont_loss_weight_multiplier
        recons_loss = F.mse_loss(recons, input)                
        loss = recons_loss + cont_loss_weighted
        return {
            "loss": loss,
            "cont_loss" : cont_loss_weighted.detach(),
            "recons_loss" : recons_loss.detach()                
        }

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

    