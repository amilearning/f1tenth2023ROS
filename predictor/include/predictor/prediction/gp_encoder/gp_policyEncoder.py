import torch 
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Type, List
from predictor.common.pytypes import VehicleState, ParametricPose, VehicleActuation, VehiclePrediction
from torch.utils.tensorboard import SummaryWriter
from barcgp.prediction.gp_encoder.gp_encoderModel import GPLSTMAutomodel
from predictor.prediction.dyn_prediction_model import TorchDynamicsModelForPredictor
from predictor.common.tracks.radius_arclength_track import RadiusArclengthTrack
import secrets

from barcgp.h2h_configs import *
from barcgp.common.utils.file_utils import *
import gpytorch 
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

writer = SummaryWriter(flush_secs=1)

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



class GPContPolicyEncoder:
        def __init__(self,means_y = None, stds_y = None, train_loader_= None, test_loader_ = None, args = None, model_load = False, model_id = 100):
            
            self.train_loader = train_loader_
            self.test_loader = test_loader_

            if means_y is not None:
                self.means_y = means_y
            else:
                self.means_y = 0.0
            if stds_y is not None:                
                self.stds_y  = stds_y
            else:
                self.stds_y = 1.0
            self.model_id = model_id    
            self.n_epochs = 1000

            if args is None:
                self.train_args = {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 9,
                "output_size": 3,
                "hidden_size": 8,
                "latent_size": 3,
                "gp_input_size": 8, ## 5 + latent_size 
                "seq_len": 5                
                }
            else: 
                self.train_args = args      
            
            self.input_dim = self.train_args["input_size"]
            self.output_dim = 3
            self.seq_len = self.train_args["seq_len"]
            
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim)  
            self.model = GPLSTMAutomodel(self.train_args).to(device='cuda')                
            lr = 0.1
            self.optimizer = SGD([
                {'params' :self.model.lstm_enc.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.lstm_dec.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.fc_l2l.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.fc21.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.relu.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.fc22.parameters(), 'weight_decay': 1e-4},
                {'params' :self.model.bn1.parameters(), 'weight_decay': 1e-4},
                {'params': self.model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
                {'params': self.model.gp_layer.variational_parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
            

            if model_load:
                self.model_load()

        def outputToReal(self, output, normalized = False):
            if normalized:
                return output
            if self.means_y is not None:
                return output * self.stds_y + self.means_y
            else:
                return output

        def reset_args(self,args):
            self.train_args = args
            self.input_dim = args["input_size"]
            self.output_dim = args["latent_size"]
            self.seq_len = args["seq_len"]

        
       

        def model_save(self,model_id= None):
            if model_id is None:
                model_id = self.model_id
            # save_dir = os.path.join(model_dir, 'cont_encoder_{model_id}.model')
            save_dir = model_dir+"/"+f"gp_encoder_{model_id}.model" 
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_args': self.train_args,
            "liklihood" : self.likelihood,
            "means_y": self.means_y,
            "stds_y": self.stds_y,
                }, save_dir)
                
            # torch.save(self.model.state_dict(), save_dir )
            print("model has been saved in "+ save_dir)

        def model_load(self,model_id =None):
            if model_id is None:
                model_id = self.model_id
            saved_data = torch.load(model_dir+"/"+f"gp_encoder_{model_id}.model")            
            loaded_args= saved_data['train_args']
            self.reset_args(loaded_args)

            model_state_dict = saved_data['model_state_dict']
            
            self.model = GPLSTMAutomodel(self.train_args).to(device='cuda')                
            self.model.to(torch.device("cuda"))
            self.model.load_state_dict(model_state_dict)
            self.likelihood = saved_data['liklihood']
            self.means_y = saved_data['means_y']
            self.stds_y = saved_data['stds_y']
            self.model.eval()            
            self.likelihood.eval()

            

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

        def get_gp_output(self,x):
            # shape of x - > [sample, sequence, input_dim]
            # shape of y - > [sample, output_dim]
            self.model.eval()
            self.likelihood.eval()
            self.model.gp_layer.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if len(x.shape) < 3:                    
                    _, _, _, _, prediction = self.model(x.unsqueeze(0))
                else:
                    _, _, _, _, prediction = self.model(x)
                mx = prediction.mean
                std = prediction.stddev
                noise = torch.cuda.FloatTensor(std.shape[0], std.shape[1]).normal_().to(device="cuda")
                tmp_target_pred = mx + noise * std  
                result = self.outputToReal(tmp_target_pred)
            return result

        def train(self,epoch , args = None):
            if self.train_loader is None:
                print(" train iterator none")
                return 
            if args is None:
                args = self.train_args
            
        
            self.likelihood = self.likelihood.cuda()
            self.model = self.model.cuda()
            self.model.gp_layer = self.model.gp_layer.cuda()

            self.model.train()
            self.likelihood.train()
            self.model.gp_layer.train()

            self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer, num_data=len(self.train_loader))

            self.train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )
            # minibatch_iter = tqdm.notebook.tqdm(self.train_loader, desc=f"(Epoch {epoch}) Minibatch")
            with gpytorch.settings.num_likelihood_samples(8):
                for i, [data, target] in self.train_iterator:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                        
                    self.optimizer.zero_grad()
                    mloss, recon_x, cont_loss, recon_loss, output = self.model(data)
                    liklihood_loss = -self.mll(output, target) 
                    loss =liklihood_loss+ mloss*1.0
                    loss.backward()
                    self.optimizer.step()
                    

                    self.train_iterator.set_postfix({"encoder_part_loss": float(mloss.mean())})                    
                    writer.add_scalar("encoder_part_loss", float(mloss.mean()), epoch)                
                    writer.add_scalar("cont_loss", float(cont_loss), epoch)        
                    writer.add_scalar("recon_loss", float(recon_loss), epoch)   
                    writer.add_scalar("loss_total", float(loss), epoch)   
                    writer.add_scalar("liklihood_loss", float(liklihood_loss), epoch)   


        def test(self,epoch):
            self.model.eval()
            self.likelihood.eval()
            correct = 0
            self.test_iterator = tqdm(
                enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
            )
            eval_loss = 0 
            cont_eval_loss = 0
            recon_eval_loss = 0
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(16):
                for i, [data, target] in self.test_iterator:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                        mloss, recon_x, cont_loss, recon_loss, output= self.model(data)
                        eval_loss += mloss.mean().item()
                        cont_eval_loss += cont_loss.mean().item()
                        recon_eval_loss += recon_loss.mean().item()
                        self.test_iterator.set_postfix({"eval_loss": float(mloss.mean())})                        
                        gp_eval_loss = -self.mll(output, target)
                eval_loss = gp_eval_loss / len(self.test_loader)
                writer.add_scalar("eval_gp_loss", float(gp_eval_loss), epoch)         
                writer.add_scalar("eval_cont_loss", float(cont_loss), epoch)         
                writer.add_scalar("eval_recon_loss", float(recon_loss), epoch)         

            

        def model_training(self, args = None):
            scheduler = MultiStepLR(self.optimizer, milestones=[0.5 * self.n_epochs, 0.75 * self.n_epochs], gamma=0.1)

            for epoch in range(1, self.n_epochs + 1):                
                with gpytorch.settings.use_toeplitz(False):
                    self.train(epoch, args)
                    self.test(epoch)
                scheduler.step()
                state_dict = self.model.state_dict()
                likelihood_state_dict = self.likelihood.state_dict()
            torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')

        
            # self.model = model
            
            # if epoch%500 == 0:
            #     self.model_save(model_id=epoch)

        def get_theta_from_buffer(self,input_for_encoder):      
            if len(input_for_encoder.shape) <3:
                input_for_encoder = input_for_encoder.unsqueeze(dim=0).to(device="cuda")
            else:
                input_for_encoder = input_for_encoder.to(device="cuda")
            theta = self.get_theta(input_for_encoder)
            
            return theta.squeeze()
        
      

        def tsne_evaluate(self):            
            if self.train_loader is None:
                return 
            args = self.train_args
            
            ## training
            count = 0
            train_iterator = tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader), desc="training"
                )
            model = self.model 
            model.eval()
            z_tmp_list = []
            input_list = []
            with torch.no_grad():
                for i, batch_data in train_iterator:    
                                
                    count += 1
                    train_data = batch_data.to(args['device'])                
                    z_tmp = model.get_latent_z(train_data)
                    if z_tmp.shape[0] == args['batch_size']:
                        z_tmp_list.append(z_tmp)
                        input_list.append(train_data)
                stacked_z_tmp = torch.cat(z_tmp_list, dim=0)
                input_list_tmp= torch.cat(input_list, dim=0)

            return stacked_z_tmp, input_list_tmp


               

        def get_true_prediction_par(self, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                                    ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=10):
        
            
            # Set GPEncoder model to eval-mode
            self.model.eval()
            self.likelihood.eval()
            # draw M samples        
            preds = self.sample_traj_gp_par(encoder_input, ego_state, target_state, ego_prediction, track, M)        
            # numeric mean and covariance calculation.
            # cov_start_time = time.time()
            pred = self.mean_and_cov_from_list(preds, M) 
            # cov_end_time = time.time()
            # cov_elapsed_time = cov_end_time - cov_start_time
            # print(f"COV Elapsed time: {cov_elapsed_time} seconds")    
            pred.t = ego_state.t
            return pred

        
        def mean_and_cov_from_list(self, l_pred: List[VehiclePrediction], M):
            """
            Extracts sample mean trajectory and covariance from list of VehiclePredictions
            """
            mean = l_pred[0].copy()
            mean.sey_cov = []
            for i in range(len(mean.s)):
                mean.s[i] = np.average([k.s[i] for k in l_pred])
                mean.x_tran[i] = np.average([k.x_tran[i] for k in l_pred])
                cov1 = np.sqrt(np.sum([(mean.s[i] - k.s[i]) ** 2 for k in l_pred]) / (M - 1))
                cov2 = np.sqrt(np.sum([(mean.x_tran[i] - k.x_tran[i]) ** 2 for k in l_pred]) / (M - 1))
                mean.sey_cov.append(np.array([[cov1, 0], [0, cov2]])[:2, :2].flatten())
                mean.e_psi[i] = np.average([k.e_psi[i] for k in l_pred])
                mean.v_long[i] = np.average([k.v_long[i] for k in l_pred])

            mean.s = array.array('d', mean.s)
            mean.x_tran = array.array('d', mean.x_tran)
            mean.sey_cov = array.array('d', np.array(mean.sey_cov).flatten())
            mean.e_psi = array.array('d', mean.e_psi)
            mean.v_long = array.array('d', mean.v_long)
            return mean


        def sample_traj_gp_par(self, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                            ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M):

            
            prediction_samples = []
            for j in range(M):
                tmp_prediction = VehiclePrediction() 
                tmp_prediction.s = []
                tmp_prediction.x_tran = []
                tmp_prediction.e_psi = []
                tmp_prediction.v_long = []              
                tmp_prediction.v_tran = []
                tmp_prediction.psidot = []      
                prediction_samples.append(tmp_prediction)

                
                
                    
            # roll state is ego and tar vehicle dynamics stacked into a big matrix 
            init_tar_state = torch.tensor([target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long, target_state.v.v_tran, target_state.w.w_psi])
            init_tar_curv = torch.tensor([target_state.lookahead.curvature[0], target_state.lookahead.curvature[1] , target_state.lookahead.curvature[2]])
            init_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long, ego_state.v.v_tran, ego_state.w.w_psi])
            init_ego_curv = torch.tensor([ego_state.lookahead.curvature[0] ,ego_state.lookahead.curvature[1] ,ego_state.lookahead.curvature[2]])
            
            ## tar (6) + ego (6) + tar_curv(3) + ego_curv(3)
            init_state = torch.hstack([init_tar_state,init_ego_state, init_tar_curv, init_ego_curv])

            roll_state = init_state.repeat(M,1).clone().to(device="cuda")        
            
            for j in range(M):                          # tar 0 1 2 3 4 5       #ego 6 7 8 9 10 11
                prediction_samples[j].s.append(roll_state[j,0].cpu().numpy())
                prediction_samples[j].x_tran.append(roll_state[j,1].cpu().numpy())                    
                prediction_samples[j].e_psi.append(roll_state[j,2].cpu().numpy())
                prediction_samples[j].v_long.append(roll_state[j,3].cpu().numpy())
                prediction_samples[j].v_tran.append(roll_state[j,4].cpu().numpy())
                prediction_samples[j].psidot.append(roll_state[j,5].cpu().numpy())                 

            roll_encoder_input = encoder_input.clone().to(device="cuda")  
            
            horizon = len(ego_prediction.x)       
            stacked_roll_state = roll_state.repeat(horizon,1).to(device="cuda")  
            for i in range(horizon-1):  
                
            ##################################################################################
            ################################## Theta prediction ##############################                  
                # encoder_start_time = time.time()
                if i >0:                
                    if len(roll_encoder_input.shape) < 3:                    
                        roll_encoder_input = roll_encoder_input.repeat(M,1,1)
                    roll_encoder_input = self.roll_encoder_input_given_rollstate(roll_encoder_input.clone(),roll_state)
                
            ################################## Target input prediction ##############################          
                # gp_start_time = time.time()
                prediction = self.get_gp_output(roll_encoder_input)
                # target_output_residual = self.outputToReal(tmp_target_pred)
                target_output_residual = prediction
                
                predicted_target_vel = roll_state[:,3:6] 
                    
                if ego_prediction.u_a is None:
                    tmp_ego_input = torch.tensor([0.0, 0.0]).to(device="cuda")    
                else:
                    tmp_ego_input = torch.tensor([ego_prediction.u_a[i], ego_prediction.u_steer[i]]).to(device="cuda")
            ################################## Target input prediction END ###########################        
            
            ################################## Vehicle Dynamics Update #################################             
                vehicle_simulator = TorchDynamicsModelForPredictor(track)   
                stacked_roll_state_for_dynamics_ego = self.roll_state_to_stack_tensor(roll_state)            
                stacked_ego_tar_roll_state = torch.vstack([roll_state[:,0:3], roll_state[:,6:9]])
                stacked_ego_tar_vel = torch.vstack([predicted_target_vel, roll_state[:,9:12]])
                next_x_ego, next_cur_ego=  vehicle_simulator.kinematic_update(roll_state[:,6:9],roll_state[:,9:12])               
                next_x_tar, next_cur_tar=  vehicle_simulator.residual_state_update(roll_state[:,0:3],roll_state[:,3:6] , target_output_residual)
                ########################
                # next_x_tar_ego, next_cur_tar_ego=  vehicle_simulator.kinematic_update(stacked_ego_tar_roll_state,stacked_ego_tar_vel)               
                # next_x_tar = next_x_tar_ego[:roll_state.shape[0],:]
                # next_x_ego = next_x_tar_ego[roll_state.shape[0]:,:]                
                # next_cur_tar = next_cur_tar_ego[0,:roll_state.shape[0]]
                # next_cur_ego = next_cur_tar_ego[0,roll_state.shape[0]:]
                ########################

                # next_x_tar, next_cur_tar=  vehicle_simulator.kinematic_update(roll_state[:,0:3],predicted_target_vel)               
                # next_x_ego, next_cur_ego = vehicle_simulator.dynamics_update(stacked_roll_state_for_dynamics_ego,tmp_ego_input.repeat(M,1).to(device="cuda"))
            
                roll_state = self.stack_tensor_to_roll_state(next_x_tar,next_x_ego,next_cur_tar,next_cur_ego)
                
            ################################## Vehicle Dynamics Update END #################################
                for j in range(M):                                
                    prediction_samples[j].s.append(roll_state[j,0].cpu().numpy())
                    prediction_samples[j].x_tran.append(roll_state[j,1].cpu().numpy())                    
                    prediction_samples[j].e_psi.append(roll_state[j,2].cpu().numpy())
                    prediction_samples[j].v_long.append(roll_state[j,3].cpu().numpy())
                    prediction_samples[j].v_tran.append(roll_state[j,4].cpu().numpy())
                    prediction_samples[j].psidot.append(roll_state[j,5].cpu().numpy())
            ################################## Vehicle Dynamics Update END##############################
                # current_states: ego_s(0), ego_ey(1), ego_epsi(2), ego_vx(3), ego_vy(4), ego_wz(5), 
                #           tar_s(6), tar_ey(7), tar_epsi(8), tar_vx(9), tar_vy(10), tar_wz(11)
                # u(0) = ax_ego, u(1) = delta_ego   
                
                
            for i in range(M):
                prediction_samples[i].s = array.array('d', prediction_samples[i].s)
                prediction_samples[i].x_tran = array.array('d', prediction_samples[i].x_tran)
                prediction_samples[i].e_psi = array.array('d', prediction_samples[i].e_psi)
                prediction_samples[i].v_long = array.array('d', prediction_samples[i].v_long)
                prediction_samples[i].v_tran = array.array('d', prediction_samples[i].v_tran)
                prediction_samples[i].psidot = array.array('d', prediction_samples[i].psidot)            

            
            
            return prediction_samples



        def roll_state_to_stack_tensor(self,roll_state):
            stacked_roll_state_for_dynamics = torch.zeros(roll_state.shape[0],6).to(device="cuda")
            # target state
            # stacked_roll_state_for_dynamics[:roll_state.shape[0],:] = roll_state[:,0:6]        
            # ego state
            stacked_roll_state_for_dynamics = roll_state[:,6:12]        
            return stacked_roll_state_for_dynamics


        def stack_tensor_to_roll_state(self,next_x_tar,next_x_ego,next_cur_tar,next_cur_ego):
            roll_state = torch.zeros(next_x_tar.shape[0],18).to(device="cuda")
            # target state
            roll_state[:,0:6] = next_x_tar
            # ego state
            roll_state[:,6:12] = next_x_ego
            # tar_curvs
            roll_state[:,12] = next_cur_tar#.squeeze()
            roll_state[:,13] = next_cur_tar#.squeeze()
            roll_state[:,14] = next_cur_tar#.squeeze()
            roll_state[:,15] = next_cur_ego#.squeeze()
            roll_state[:,16] = next_cur_ego#.squeeze()
            roll_state[:,17] = next_cur_ego#.squeeze()
            
            # roll_state[:,12] = next_cur_tar[0].squeeze()
            # roll_state[:,13] = next_cur_tar[1].squeeze()
            # roll_state[:,14] = next_cur_tar[2].squeeze()
            # roll_state[:,15] = next_cur_ego[0].squeeze()
            # roll_state[:,16] = next_cur_ego[1].squeeze()
            # roll_state[:,17] = next_cur_ego[2].squeeze()
            return roll_state

        def rollstate_to_vehicleState(self,tar_state,ego_state,tar_input, ego_input,roll_state):
            tar_state.p.s = np.mean(roll_state[:,0].cpu().numpy())
            tar_state.p.x_tran = np.mean(roll_state[:,1].cpu().numpy())
            tar_state.p.e_psi = np.mean(roll_state[:,2].cpu().numpy())
            tar_state.v.v_long = np.mean(roll_state[:,3].cpu().numpy())
            tar_state.v.v_tran = np.mean(roll_state[:,4].cpu().numpy())
            tar_state.w.w_psi = np.mean(roll_state[:,5].cpu().numpy())       
            # tar_state.lookahead.curvature[0] =  np.mean(roll_state[:,5].cpu().numpy())     
    
            ego_state.p.s = np.mean(roll_state[:,6].cpu().numpy())
            ego_state.p.x_tran = np.mean(roll_state[:,7].cpu().numpy())
            ego_state.p.e_psi = np.mean(roll_state[:,8].cpu().numpy())
            ego_state.v.v_long = np.mean(roll_state[:,9].cpu().numpy())
            ego_state.v.v_tran = np.mean(roll_state[:,10].cpu().numpy())
            ego_state.w.w_psi = np.mean(roll_state[:,11].cpu().numpy())            
    

        def vehicleState_to_rollstate(self,tar_state,ego_state,tar_input, ego_input,roll_state):
            roll_state[:,0] = tar_state.p.s
            roll_state[:,1] = tar_state.p.x_tran
            roll_state[:,2] = tar_state.p.e_psi
            roll_state[:,3] = tar_state.v.v_long
            roll_state[:,4] = tar_state.v.v_tran
            roll_state[:,5] = tar_state.w.w_psi
            roll_state[:,6] = ego_state.p.s
            roll_state[:,7] = ego_state.p.x_tran
            roll_state[:,8] = ego_state.p.e_psi
            roll_state[:,9] = ego_state.v.v_long
            roll_state[:,10] = ego_state.v.v_tran
            roll_state[:,11] = ego_state.w.w_psi
            roll_state[:,12] = tar_state.lookahead.curvature[0]
            roll_state[:,13] = tar_state.lookahead.curvature[1]
            roll_state[:,14] = tar_state.lookahead.curvature[2]
            roll_state[:,15] = ego_state.lookahead.curvature[0]
            roll_state[:,16] = ego_state.lookahead.curvature[1]
            roll_state[:,17] = ego_state.lookahead.curvature[2]
        
        def roll_encoder_input_given_rollstate(self,encoder_input,rollstate):
            # [batch, sequence, #features]
            encoder_input_tmp = encoder_input.clone()
            encoder_input_tmp[:,0:-1,:] = encoder_input[:,1:,:]
            encoder_input_tmp[:,-1,0] = rollstate[:,0] - rollstate[:,6] # tar_s.p.s - ego_s.p.s 
            encoder_input_tmp[:,-1,1:4] = rollstate[:,1:4]  # tar_st.p.x_tran, tar_st.p.e_psi, tar_st.v.v_long
            encoder_input_tmp[:,-1,4] = rollstate[:,12]  # tar_st.lookahead.curvature[0],
            encoder_input_tmp[:,-1,5:8] = rollstate[:,7:10] # ego_st.p.x_tran, ego_st.p.e_psi,ego_st.v.v_long,                       
            encoder_input_tmp[:,-1,8] = rollstate[:,15] # ego_st.lookahead.curvature[0]
            return encoder_input_tmp
        
        ## TODO: This should be matched with the below funciton 
        # states_to_encoder_input_torch
    

            