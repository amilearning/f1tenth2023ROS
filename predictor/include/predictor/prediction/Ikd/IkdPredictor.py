import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from barcgp.prediction.Ikd.IkdModels import CVAE, FFNN, ConvFFNN
from torch.utils.tensorboard import SummaryWriter
import secrets

from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *




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



class IkdPredictor:
        def __init__(self,args = None, model_type = "ConvFFNN",model_load = False, model_id = 100):
            self.train_data = None           
            self.model = None
            self.train_loader = None
            self.test_loader = None
            self.model_id = model_id                    
            self.model_type = model_type
            self.writer = SummaryWriter()
            
            
            if args is None:
                self.train_args =  {
                    "batch_size": 128,
                    "device": torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu"),
                    "input_size": 10,
                    "hidden_size": 20,
                    "latent_size": 2,
                    "output_size": 2,
                    "learning_rate": 0.001,
                    "max_iter": 30000,
                    "sequence_length": 5
                }
            else: 
                self.train_args = args      
            
            if model_load:
                self.model_load()
            
            self.ego_state_buffer = []
            self.tar_state_buffer = []
            self.buffer_length = self.train_args["sequence_length"]*2 ## 
            self.ego_state_buffer_torch = torch.zeros(self.buffer_length,self.train_args["input_size"]+self.train_args["output_size"])
            self.tar_state_buffer_torch = torch.zeros(self.buffer_length,self.train_args["input_size"]+self.train_args["output_size"])
            self.buffer_update_count = 0

            self.ego_buffer_for_encoder = None            
            self.tar_buffer_for_encoder = None   

      
            
                     
            
        
        #     if i == t+self.target_horizon-1:
        #         target_st = scenario_data.ego_states[i]                                    
        #     ego_st = scenario_data.ego_states[i]
        #     ego_st_next = scenario_data.ego_states[i+1]                                
        #     del_s = ego_st_next.p.s - ego_st.p.s
        #     del_ey = ego_st_next.p.x_tran - ego_st.p.x_tran
        #     del_epsi = ego_st_next.p.e_psi - ego_st.p.e_psi
        #     curvature = ego_st.lookahead.curvature[0]
        #     dat[:,i-t] = torch.tensor([del_s, del_ey,del_epsi, curvature])
        
        # self.input_data.append(dat)                            
        # output = torch.tensor([target_st.u.u_a, target_st.u.u_steer])
        
            # [(tar_s-ego_s),
        #  ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
        #  tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta] 
        def append_vehicleState(self,ego_state: VehicleState,tar_state: VehicleState):     
            ######################### rolling into buffer ##########################
            # encoder_input = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.lookahead.curvature[0]])
            tmp_target_state = torch.tensor([tar_state.p.s, tar_state.p.x_tran, tar_state.p.e_psi, tar_state.lookahead.curvature[0], tar_state.u.u_a, tar_state.u.u_steer])            
            tmp_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.lookahead.curvature[0], ego_state.u.u_a, ego_state.u.u_steer])            
            tar_stacked = torch.vstack([self.tar_state_buffer_torch, tmp_target_state])
            ego_stacked = torch.vstack([self.ego_state_buffer_torch, tmp_ego_state])
            self.tar_state_buffer_torch = tar_stacked[1:,:]
            self.ego_state_buffer_torch = ego_stacked[1:,:]
            self.buffer_update_count +=1
            
            if self.buffer_update_count > 10 and self.model is not None:
                self.buffer_to_IkdInput()

        def buffer_to_IkdInput(self):
            del_torch = self.tar_state_buffer_torch[1:,:4] - self.tar_state_buffer_torch[0:-1,:4]             
            del_torch[:,-1] = self.tar_state_buffer_torch[1:,3]
            seq_length = self.train_args["sequence_length"]
            # Extract and stack every sub-tensor of shape [seq_length, 6]
            sub_tensors = []
            for i in range(del_torch.shape[0] - seq_length + 1):
                sub_tensor = del_torch[i:i+seq_length, :]
                sub_tensors.append(sub_tensor)

            buffer_to = torch.stack(sub_tensors)
            est_tar_action = self.model.predict(buffer_to)
            ######################### Very important for matching with the training data 
            input_step_back_idx = 2 # --> sampleGeneroator.input_step_back_idx
            ################################################################################
            start_idx = self.tar_state_buffer_torch.shape[0]-input_step_back_idx-est_tar_action.shape[0]-1
            end_idx = self.tar_state_buffer_torch.shape[0]-input_step_back_idx-1
            ### Overise the Inverse kinodynamics based input estimation 
            # self.tar_state_buffer_torch[start_idx:end_idx,-2:] = est_tar_action
            
            ### get ready the input for Autoencoder 
            self.ego_buffer_for_encoder = self.tar_state_buffer_torch[start_idx:end_idx,:]
            self.tar_buffer_for_encoder = self.ego_state_buffer_torch[start_idx:end_idx,:]
            
            


        def set_train_loader(self,data_loader):
            self.train_loader = data_loader

        def set_test_loader(self,data_loader):
            self.test_loader = data_loader

        def set_train_data(self,x_data, y_data):            
            train_dataset = MyDataset(x_data, y_data)            
            self.train_loader = DataLoader(train_dataset, batch_size=self.train_args["batch_size"], shuffle=True)
        
        def set_test_data(self,x_data,y_data):
            test_dataset = MyDataset(x_data, y_data)            
            self.test_loader = DataLoader(test_dataset, batch_size=self.train_args["batch_size"], shuffle=False)

        def model_save(self,model_id= None):
            if model_id is None:
                model_id = self.model_id
            save_dir = model_dir+f"ikd_{model_id}.model"
            print("model has been saved in "+ save_dir)

            torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'train_args': self.train_args
                         }, save_dir)


            # torch.save(self.model.state_dict(), save_dir )

        def model_load(self,model_id =None):
            if model_id is None:
                model_id = self.model_id

            saved_data = torch.load(model_dir+f"ikd_{model_id}.model")
            
            self.train_args = saved_data['train_args']
            model_state_dict = saved_data['model_state_dict']

            if self.model_type == "CVAE":
                self.model = CVAE(self.train_args)
            elif self.model_type == "FFNN":
                self.model = FFNN(self.train_args)
            elif self.model_type == "ConvFFNN":
                self.model = ConvFFNN(self.train_args)                            
            self.model.to(torch.device("cuda"))
            self.model.load_state_dict(model_state_dict)
            self.model.eval()       

            # self.model.load_state_dict(torch.load(model_dir+f"ikd_{model_id}.model"))
                 
                
        def train(self,args = None):
            if self.train_loader is None:
                return 
            if args is None:
                args = self.train_args
            else:
                self.train_args = args 

            if self.model is None:                
                if self.model_type == "CVAE":
                    model = CVAE(args).to(device='cuda')
                elif self.model_type == "FFNN":
                    model = FFNN(args).to(device='cuda')
                elif self.model_type == "ConvFFNN":
                    model = ConvFFNN(args).to(device='cuda')
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
                
                for i, (batch_x,batch_y) in train_iterator:

                    if count > args['max_iter']:
                        self.model = model
                        return model
                    count += 1                    
                    mloss, recon_x, info, theta_mean_, theta_logvar_ = model(batch_x.float(), batch_y.float(), batch_x, batch_y,infererence = False)
                    # Backward and optimize
                    optimizer.zero_grad()
                    mloss.mean().backward()
                    optimizer.step()

                    train_iterator.set_postfix({"train_loss": float(mloss.mean())})
                self.writer.add_scalar("train_loss", float(mloss.mean()), epoch)
            
            self.model = model
            self.model.eval()
            
            print("IKD predictor train done")

        def draw_eval(self, gt,est,y_gt):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            gt_torch = torch.concat(gt)            
            est_torch = torch.concat(est)
            y_gt_torch = torch.concat(y_gt)

            gt_np = gt_torch.cpu().numpy()
            est_np = est_torch.cpu().numpy()
            y_gt_np = y_gt_torch.cpu().numpy()
            
            # Plot the first column of A against the first column of B
            ax1.plot(y_gt_np[:,0],'*')
            ax1.plot(est_np[:,0])
            ax2.plot(y_gt_np[:,1],'*')
            ax2.plot(est_np[:,1])
            # Adjust the spacing between the subplots
            plt.subplots_adjust(wspace=0.4)
            # Show the plot
            plt.show()
            aa = 2

        def evaluate(self):        
            if self.test_loader is not None:                    
                self.model.eval()
                eval_loss = 0
                test_iterator = tqdm(
                    enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
                )
                
                x_gt_list = []
                recon_list = []
                y_gt_list = []
                with torch.no_grad():
                    for i, (batch_x,batch_y) in test_iterator:                        
                     
                        mloss, recon_x, (recon_loss, kld_loss), theta_mean_, theta_logvar_ = self.model(batch_x,batch_y,batch_x,batch_y, infererence = True)
                        x_gt_list.append(batch_x)
                        recon_list.append(recon_x)
                        y_gt_list.append(batch_y)
                        eval_loss += mloss.mean().item()

                        test_iterator.set_postfix({"eval_loss": float(mloss.mean())})


                eval_loss = eval_loss / len(self.test_loader)
                self.writer.add_scalar("eval_loss", float(eval_loss))
                print("Evaluation Score : [{}]".format(eval_loss))
                self.draw_eval(x_gt_list,recon_list,y_gt_list)
