import torch
from torch import nn 
from torch.nn import functional as F
from barcgp.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate
import gpytorch
    
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
        

class GPLSTMAutomodel(gpytorch.Module):    
    """LSTM-based Contrasiave Auto Encoder"""
    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(GPLSTMAutomodel, self).__init__()        
        
        self.device = args['device']
        # dimensions
        self.input_size = args['input_size']
        self.output_size = args['output_size']
        self.hidden_size = args['hidden_size']
        self.latent_size = args['latent_size']
        self.gp_input_size = args['gp_input_size']

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
        self.bn1 = nn.BatchNorm1d(num_features=self.latent_size)           
        
        
        self.gp_layer = IndependentMultitaskGPModelApproximate(inducing_points_num=200,
                                                                input_dim=self.gp_input_size,
                                                                num_tasks=self.output_size)  # Independent
        

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
        ############# theta is latent varaible
        z = self.bn1(z)
        theta = z.clone()
        # features = theta.transpose(-1, -2).unsqueeze(-1)
        # last step [tar_ey, tar_epsi, tar_vlong, cur_0, cur_2]
        # tar_ey = x[:,-1,1]
        # tar_epsi = x[:,-1,2]
        # tar_vlong = x[:,-1,3]
        # curvature[0] = x[:,-1,4]        
        # curvature[2] = x[:,-1,-1]
        
        curvature2 = torch.unsqueeze(x[:,-1,-1], dim=1)        
        gp_input = torch.cat((x[:,-1,1:5], curvature2,theta), dim=1)

        gpoutput = self.gp_layer(gp_input)
        
        ######################################### The end of GP branch 

        z = self.fc_l2l(z.to(self.device))        
        # decode latent space to input space
        # z = z.repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
        # reconstruct_output, hidden = self.lstm_dec(z, enc_hidden)
        reconstruct_output, hidden = self.lstm_dec(z)
        # reconstruct_output, hidden = self.lstm_dec(z)
        x_hat = reconstruct_output
        
        # calculate  reconstructionloss
        losses = self.loss_cont(x_hat, x, theta)
        
       
        ####################
        m_loss = losses["encoder_part_loss"]
        cont_loss = losses["cont_loss"]
        recon_loss = losses["recons_loss"]
        
        return m_loss, x_hat, cont_loss, recon_loss, gpoutput
                
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
        # normalized_tensor = F.normalize(seq_sum_distances, dim=(0, 1))
        return seq_sum_distances   
       
    def square_exponential_kernel(self, x1, x2, length_scale=1.7, variance=0.1):
        """
        Computes the square exponential (RBF) kernel between two sets of data points.

        Args:
            x1 (Tensor): First set of data points with shape (N1, D).
            x2 (Tensor): Second set of data points with shape (N2, D).
            length_scale (float, optional): Length scale parameter. Default is 1.0.
            variance (float, optional): Variance parameter. Default is 1.0.

        Returns:
            kernel_matrix (Tensor): Kernel matrix with shape (N1, N2).
        """
        diff = x1.unsqueeze(0) - x2.unsqueeze(1)
        norm_sq = torch.sum(diff ** 2, dim=-1)        
        kernel_matrix = variance * torch.exp(-0.5 * norm_sq / (length_scale ** 2))
        return kernel_matrix

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
        theta_diff_mean = self.square_exponential_kernel(theta,theta) # theta_diff_mean = batch x batch 
        upper_bound = 1.3* input_diff_mean
        lower_bound = 0.7* input_diff_mean
        deviation_loss = torch.sum(torch.relu(theta_diff_mean - upper_bound) + torch.relu(lower_bound - theta_diff_mean))

        cont_loss = deviation_loss # F.mse_loss(input_diff_mean,theta_diff_mean)
        
        
        recons_loss = F.mse_loss(recons, input)                

        cont_loss_weight_multiplier = 1e-5
        cont_loss_weighted = cont_loss*cont_loss_weight_multiplier
        recons_loss_weight = 1e-10
        recons_loss_weighted = recons_loss*recons_loss_weight
        loss = recons_loss_weighted + cont_loss_weighted
        return {
            "encoder_part_loss": loss,
            "cont_loss" : cont_loss_weighted.detach(),
            "recons_loss" : recons_loss_weighted.detach()                
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

    