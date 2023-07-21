import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
from predictor.prediction.encoder.encoderModel import LSTMAutomodel
import secrets

from predictor.h2h_configs import *
from predictor.common.utils.file_utils import *

# writer = SummaryWriter()

class MyDataset:
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
       
        return x, y



class PolicyEncoder:
        def __init__(self,args = None,model_load = False, model_id = 100):
            self.train_data = None           
            self.model = None
            self.train_loader = None
            self.test_loader = None
            self.model_id = model_id                                
            
            if args is None:
                self.train_args = {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 9,
                "hidden_size": 8,
                "latent_size": 4,
                "seq_len": 5,
                "learning_rate": 0.0001,
                "max_iter": 60000,
                }
            else: 
                self.train_args = args      
            
            self.input_dim = self.train_args["input_size"]
            self.output_dim = self.train_args["latent_size"]
            self.seq_len = self.train_args["seq_len"]
            
            if model_load:
                self.model_load()
                     
        def reset_args(self,args):
            self.train_args = args
            self.input_dim = args["input_size"]
            self.output_dim = args["latent_size"]
            self.seq_len = args["seq_len"]

        
        def set_train_loader(self,data_loader):
            self.train_loader = data_loader

        def set_test_loader(self,data_loader):
            self.test_loader = data_loader
            

        def model_save(self,model_id= None):
            if model_id is None:
                model_id = self.model_id
            save_dir = model_dir+f"encoder_{model_id}.model" 
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_args': self.train_args
                }, save_dir)
                
            # torch.save(self.model.state_dict(), save_dir )
            print("model has been saved in "+ save_dir)

        def model_load(self,model_id =None):
            if model_id is None:
                model_id = self.model_id
            saved_data = torch.load(model_dir+f"encoder_{model_id}.model")            
            loaded_args= saved_data['train_args']
            self.reset_args(loaded_args)

            model_state_dict = saved_data['model_state_dict']
            self.model = LSTMAutomodel(self.train_args).to(device='cuda')                
            self.model.to(torch.device("cuda"))
            self.model.load_state_dict(model_state_dict)
            self.model.eval()            

        def get_theta(self,x,np = False):

            z = self.model.get_latent_z(x)
            ###  For TEsting only -> if InputPredictGP is working with the ground truth theta 
            # z = torch.ones(z.shape).to(device="cuda")
            ###
            if torch.is_tensor(z) is False and np is False:                
                z = torch.tensor(z)
            elif torch.is_tensor(z) and np:        
                z = z.cpu().numpy()            
            return z


        def train(self,args = None):
            if self.train_loader is None:
                return 
            if args is None:
                args = self.train_args
            
            if self.model is None:     
                model = LSTMAutomodel(args).to(device='cuda')                           
            else:
                model = self.model.to(device='cuda')

            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

            ## interation setup
            epochs = tqdm(range(args['max_iter'] // len(self.train_loader) + 1))

            ## training
            count = 0
            for epoch in epochs:
                model.train()
                optimizer.zero_grad()
                train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )

                for i, batch_data in train_iterator:

                    if count > args['max_iter']:
                        print("count exceed")
                        self.model = model
                        return model
                    count += 1

                    train_data = batch_data.to(args['device'])

                    ## reshape
                    batch_size = train_data.size(0)
                    # example_size = past_data.size(1)
                    # image_size = past_data.size(1), past_data.size(2)
                    # past_data = (
                    #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
                    # )
                    # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)
                    # m_loss, x_hat, mean_consensus_loss
                    mloss, recon_x = model(train_data)

                    # Backward and optimize
                    optimizer.zero_grad()
                    mloss.mean().backward()
                    optimizer.step()

                    train_iterator.set_postfix({"train_loss": float(mloss.mean())})                    
                # writer.add_scalar("train_loss", float(mloss.mean()), epoch)                

                model.eval()
                eval_loss = 0
                consensus_loss = 0
                test_iterator = tqdm(
                    enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
                )

                with torch.no_grad():
                    for i, batch_data in test_iterator:
                        test_data = batch_data.to(args['device'])

                        ## reshape
                        batch_size = test_data.size(0)
                        # example_size = past_data.size(1)
                        # past_data = (
                        #     past_data.view(batch_size, example_size, -1).float().to(args['device'])
                        # )
                        # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

                        mloss, recon_x  = model(test_data)

                        eval_loss += mloss.mean().item()
                        test_iterator.set_postfix({"eval_loss": float(mloss.mean())})                        

                        # if i == 0:
                        #     for idx in range(len(recon_x)):
                        #         plt.plot(recon_x[idx,:,0].cpu(), recon_x[idx,:,0].cpu(), 'r.')
                        #         plt.plot(past_data[idx,:,0].cpu(), past_data[idx,:,0].cpu(), 'g.')                        
                        #         plt.pause(0.01)
                        #     plt.clf()
                            # nhw_orig = past_data[0].view(example_size, image_size[0], -1)
                            # nhw_recon = recon_x[0].view(example_size, image_size[0], -1)                    
                            # writer.add_images(f"original{i}", nchw_orig, epoch)
                            # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)

                eval_loss = eval_loss / len(self.test_loader)
                # writer.add_scalar("eval_loss", float(eval_loss), epoch)                
                print("Evaluation Score : [{}]".format(eval_loss))
                
                self.model = model

        def get_theta_from_buffer(self,input_for_encoder):      
            if len(input_for_encoder.shape) <3:
                input_for_encoder = input_for_encoder.unsqueeze(dim=0).to(device="cuda")
            else:
                input_for_encoder = input_for_encoder.to(device="cuda")
            theta = self.get_theta(input_for_encoder)
            
            return theta.squeeze()
        
        def states2encoderinput(self, ego_states, tar_states):       
            ## convert from vehiclestatus to input vecotr for encoder 
            # Input for encoder -> 
            # [(tar_s-ego_s),
            #  ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
            #  tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta] 
            for i in range(self.time_length):            
                s_diff = tar_states[i].p.s-ego_states[i].p.s
                # if s_diff > track_info.track_length/2.0:
                #     s_diff = s_diff-track_info.track_length 
                input_tmp = torch.tensor([s_diff,
                                                ego_states[i].p.x_tran,
                                                bound_angle_within_pi(ego_states[i].p.e_psi),
                                                ego_states[i].lookahead.curvature[1],                                      
                                                ego_states[i].u.u_a,
                                                ego_states[i].u.u_steer,
                                                tar_states[i].p.x_tran,
                                                bound_angle_within_pi(tar_states[i].p.e_psi),
                                                tar_states[i].lookahead.curvature[1],
                                                tar_states[i].u.u_a,
                                                tar_states[i].u.u_steer]).to(device = self.device)                                      
                if i == 0:
                    input = input_tmp
                else:
                    input = torch.vstack([input,input_tmp])
            
            return input

        def tsne_evaluate(self):
            return