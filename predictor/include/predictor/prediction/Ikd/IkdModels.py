import torch
from torch import nn 
from torch.nn import functional as F

class CVAEEncoder(nn.Module):
    def __init__(self,input_size = 8, output_size = 1, hidden_size = 5, num_latent = 1):
        super(CVAEEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_latent = num_latent
       
        self.enc = nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),  # 1 additional input for label
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_latent * 2)  # two outputs for mean and variance
        )

    def forward(self, x,y):    
        xx = torch.cat([x, y], dim=1)    
        outputs = self.enc(xx)
        return outputs

class CVAEDecoder(nn.Module):
    def __init__(
        self, input_size=8, hidden_size=5, output_size=1, num_latent=1):
        super(CVAEDecoder, self).__init__()        
        self.output_size = output_size
        self.num_latent = num_latent
        self.dec = nn.Sequential(
            nn.Linear(num_latent + input_size, hidden_size),  # 1 additional input for label
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, y):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        xx = torch.cat([x, y], dim=1)    
        output = self.dec(xx)        
        return output
        

class CVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(CVAE, self).__init__()
        self.device = args['device']

        # dimensions
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.latent_size = args['latent_size']
        self.output_size= args['output_size']
   
        # def __init__(self,input_size = 8, output_size = 1, hidden_size = 5, num_latent = 1):
        self.enc = CVAEEncoder(
            input_size=self.input_size,output_size = self.output_size, hidden_size=self.hidden_size, num_latent=self.latent_size
        )
        
        self.dec = CVAEDecoder(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_latent=self.latent_size,
        )
        

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def forward(self, xin, yin,xout = None, yout = None, infererence = False):
        # when the size of xin and y in is not equal then we sample  xin first)
        if infererence:
            xin = xin[0,:].view(-1,xin.shape[1])
            xin = xin.repeat(yout.shape[0],1)
            yin = yin[0,:].view(-1,yin.shape[1])
            yin = yin.repeat(yout.shape[0],1)
        
        batch_size, feature_dim = xin.shape

        # encode input space to hidden space
        h = self.enc(xin,yin)
        mean, logvar = torch.split(h, self.latent_size, dim=1)
       
        z = self.reparametize(mean, logvar)  # batch_size x latent_size

        
        
       
        # calculate vae loss
        if infererence:
            x_hat = self.dec(z, yout)
            losses = self.loss_function(x_hat, xout, mean, logvar)
        else:
            x_hat = self.dec(z, yin)
            losses = self.loss_function(x_hat, xin, mean, logvar)
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

        kld_weight = 0.0025  # Account for the minibatch samples from the dataset
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



class FFNN(nn.Module):
    """Feedfoward neural network"""
    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x sequence_length x input_dim
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(FFNN, self).__init__()
        self.device = args['device']
        # dimensions
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.output_size= args['output_size']
   
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),  # 1 additional input for label
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)  # two outputs for mean and variance
        )
        

    def forward(self, xin, yin,xout=None,yout=None,infererence=None):
        # when the size of xin and y in is not equal then we sample  xin first)
     
        x_hat = self.fc(yin)
        losses = self.loss_function(x_hat, xin)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )
        return m_loss, x_hat, (recon_loss, kld_loss), 0, 0


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
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": 0.0
        }


class ConvFFNN(nn.Module):
    """1dconv neural network"""
    def __init__(
        self, args):
        """
        args['input_size']: int, batch_size x  input_dim x sequence_length
        args['hidden_size']: int, output size of LSTM VAE
        args['latent_size']: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(ConvFFNN, self).__init__()
        self.device = args['device']
        # dimensions
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.output_size= args['output_size']
        self.sequence_length = args['sequence_length']
        num_filters = 20
        kernel_size = 3


       
        self.conv1d = nn.Conv1d(self.input_size, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_filters*(self.sequence_length-kernel_size+1),self.hidden_size)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)


        # self.hidden_dim = self.hidden_size
        # self.num_layers = 1
        # self.lstm = nn.LSTM( self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, 6)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(6, self.output_size)


    def predict(self,x):        
        x = x.to(device="cuda",dtype=torch.float)
        if x.shape[-1] == self.input_size:
            x = x.permute(0,2,1) 
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu2(x)
        y = self.linear2(x)
        return y



    def forward(self, x, y,xout=None,yout=None,infererence=None):
        # when the size of xin and y in is not equal then we sample  xin first)
    

        x = self.conv1d(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu2(x)
        y_hat = self.linear2(x)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(torch.device("cuda")) # initialize hidden state with zeros
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(torch.device("cuda")) # initialize cell state with zeros
        
        # x_tmp = x.permute(0,2,1)

        # out, _ = self.lstm(x_tmp) # pass the input through the LSTM layers
        # out = self.fc(out[:, -1, :]) # pass the last hidden state to the fully connected layer
        # out = self.relu(out)
        # y_hat = self.fc2(out)
        




        losses = F.mse_loss(y_hat, y)
        return losses,  y_hat, (0, 0), 0, 0
         



