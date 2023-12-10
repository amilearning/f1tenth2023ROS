import torch
from barcgp.common.utils.scenario_utils import *
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *    
from barcgp.common.utils.scenario_utils import policy_generator
from barcgp.common.utils.file_utils import *
def states_to_encoder_input_torch(tar_st,ego_st):
    tar_s = tar_st.p.s
    ego_s = ego_st.p.s
    ######### doubled track ##########
    # if tar_s > track.track_length/2.0:
    #     tar_s -=  track.track_length/2.0
    # if ego_s > track.track_length/2.0:
    #     ego_s -=  track.track_length/2.0

    delta_s = tar_s - ego_s
        

    input_data=torch.tensor([ delta_s,                        
                        tar_st.p.x_tran,
                        tar_st.p.e_psi,
                        tar_st.v.v_long,
                        tar_st.lookahead.curvature[0],
                        tar_st.lookahead.curvature[2],
                        ego_st.p.x_tran,
                        ego_st.p.e_psi, 
                        ego_st.v.v_long                       
                        ])
    return input_data




class SampleGeneartorCOVGP(SampleGenerator):
    def __init__(self, abs_path,pre_load_data_name = None, randomize=False, elect_function=None, init_all=True):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''

        # Input for Encoder -> 
        # [(tar_s-ego_s),
        #  ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
        #  tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta] 
        #   x time_horizon
        #          
        self.normalized_sample = None
        self.normalized_output = None
        self.input_dim = 9
        self.time_horizon = 10
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.output_data = []
        self.info = []
        self.means_y = None
        self.stds_y = None
        
        # if not dest_path.exists()        
        if pre_load_data_name is not None:        
            pre_data_dir =os.path.join(os.path.dirname(abs_path[0]),pre_load_data_name+'.pkl')
            self.load_pre_data(pre_data_dir)            
        else:
            pre_data_dir =os.path.join(os.path.dirname(abs_path[0]),"preload_data.pkl")                        
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        scenario_data: SimData = pickle.load(dbfile)                                
                        # scenario_data: RealData = pickle.load(dbfile)                        
                        N = scenario_data.N                       
                        ######################## random Policy ############################
                        policy_name = ab_p.split('/')[-2]
                        policy_gen = False
                        if policy_name == 'wall':
                            policy_gen = True
                            # tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.scenario_def.track)                    
                            tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.track)                    
                        ###################################################################
                        if N > self.time_horizon+5:
                            for t in range(N-1-self.time_horizon):                            
                                # define empty torch with proper size 
                                dat = torch.zeros(self.input_dim, self.time_horizon)
                                
                                for i in range(t,t+self.time_horizon):                                
                                    ego_st = scenario_data.ego_states[i]
                                    tar_st = scenario_data.tar_states[i]
                                    ntar_orin = scenario_data.tar_states[i+1]
                                    real_dt = ntar_orin.t - tar_st.t 
                                    # valid_data = self.data_validation(ego_st,tar_st,scenario_data.tar_states[i+1],scenario_data.track)                        
                                    # if valid_data and real_dt > 0.05:
                                        # dt = 0.1                        
                                        # ntar_st = interp_state_with_vel(scenario_data.track, tar_st,ntar_orin,dt).copy()



                                    if policy_gen:
                                        scenario_data.tar_states[i+1] = policy_generator(tar_dynamics_simulator,scenario_data.tar_states[i])                  
                                    # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                    #                 tar_ey, tar_epsi, tar_cur]                                 
                                    dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st)
                                    # torch.tensor([ tar_st.p.s - ego_st.p.s,
                                    #                             ego_st.p.x_tran,
                                    #                             ego_st.p.e_psi,
                                    #                             ego_st.lookahead.curvature[0],                                                            
                                    #                             tar_st.p.x_tran,
                                    #                             tar_st.p.e_psi,
                                    #                             tar_st.lookahead.curvature[0]])
                                        
                                
                                ### Add curvature[2] at the last dimension 
                                
                                next_tar_st = ntar_orin.copy()
                                real_dt = next_tar_st.t - tar_st.t 
                                valid_data = self.data_validation(ego_st,tar_st,next_tar_st,scenario_data.scenario_def.track)                        
                                if valid_data and real_dt > 0.05:
                                    # dt = 0.1                        
                                    # ntar_st = interp_state_with_vel(scenario_data.track, tar_st,next_tar_st,dt).copy()
                                    # # state_input = torch.tensor([tar_st.p.x_tran,
                                    # #                             tar_st.p.e_psi,                                                            
                                    # #                             tar_st.v.v_long,                                                           
                                    # #                             tar_st.v.v_tran,                                                           
                                    # #                             tar_st.lookahead.curvature[0],
                                    # #                             tar_st.lookahead.curvature[2]]).to(torch.device("cuda"))  
                                    # state_input = torch.tensor([   tar_st.p.x_tran,
                                    #                                 tar_st.p.e_psi,                                                            
                                    #                                 tar_st.v.v_long,                                          
                                    #                                 tar_st.lookahead.curvature[0],                                                            
                                    #                                 tar_st.lookahead.curvature[2]]).to(torch.device("cuda"))  
                                    # del_state = self.get_residual_pose_using_kinematicmodel(tar_st,next_tar_st,dt=0.1)
                                    delta_s = next_tar_st.p.s-tar_st.p.s
                                    delta_xtran = next_tar_st.p.x_tran-tar_st.p.x_tran
                                    delta_epsi = next_tar_st.p.e_psi-tar_st.p.e_psi
                                    delta_vlong  = next_tar_st.v.v_long-tar_st.v.v_long
                                    gp_output = torch.tensor([delta_s, delta_xtran, delta_epsi, delta_vlong ])                                
                                    # gp_output = torch.tensor(del_state)                                
                                    self.samples.append(dat.clone())  
                                    self.output_data.append(gp_output.clone())    
                                
                            
                        
                        dbfile.close()
            self.save_pre_data(pre_data_dir)
        
        
        self.input_output_normalizing()
        print('Generated Dataset with', len(self.samples), 'samples!')
        
        # if randomize:
        #     random.shuffle(self.samples)
    
    
    def load_pre_data(self,pre_data_dir):        
        model = pickle_read(pre_data_dir)
        self.samples = model['samples']
        self.output_data = model['output_data']    
        print('Successfully loaded data')

    def save_pre_data(self,pre_data_dir):
        model_to_save = dict()
        model_to_save['samples'] = self.samples
        model_to_save['output_data'] = self.output_data
        pickle_write(model_to_save,pre_data_dir)
        print('Successfully saved data')

    def normalize(self,data):
        mean = torch.mean(data,dim=0)
        std = torch.std(data,dim=0)        
        if len(data.shape) ==2 :
            new_data = (data - mean.repeat(data.shape[0],1))/std         
        elif len(data.shape) ==3:
            new_data = (data - mean.repeat(data.shape[0],1,1))/std         
        return new_data, mean, std

    def input_output_normalizing(self,name = 'normalizing'):
        tensor_sample = torch.stack(self.samples)
        tensor_output = torch.stack(self.output_data)
        self.normalized_sample, mean_sample, std_sample= self.normalize(tensor_sample)
        self.normalized_output, mean_output, std_output = self.normalize(tensor_output)        
        model_to_save = dict()
        model_to_save['mean_sample'] = mean_sample
        model_to_save['std_sample'] = std_sample
        model_to_save['mean_output'] = mean_output
        model_to_save['std_output'] = std_output
        pickle_write(model_to_save, os.path.join(model_dir, name + '.pkl'))
        print('Successfully saved normalizing constnats', name)
        
    
    def get_residual_pose_using_kinematicmodel(self,state,nstate, dt = 0.1):
        kinmatic_nstate = state.copy()
        vx =state.v.v_long
        vy =state.v.v_tran
        curs = state.lookahead.curvature[0]
        ey = state.p.x_tran
        epsi = state.p.e_psi
        wz = state.w.w_psi
        kinmatic_nstate.p.s = state.p.s + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) )
        kinmatic_nstate.p.x_tran = state.p.x_tran + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        kinmatic_nstate.p.e_psi = state.p.e_psi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) * curs )
        delta_state = [kinmatic_nstate.p.x_tran - nstate.p.x_tran, kinmatic_nstate.p.e_psi - nstate.p.e_psi, nstate.v.v_long - state.v.v_long]
        return delta_state
        
        
     
    def data_validation(self,ego_st: VehicleState,tar_st: VehicleState,ntar_st: VehicleState,track : RadiusArclengthTrack):
        valid_data = True
        if ego_st.p.s > track.track_length/2.0+0.5 or tar_st.p.s > track.track_length/2.0+0.5:
            valid_data = False
        
        if abs(ego_st.p.x_tran) > track.track_width or abs(tar_st.p.x_tran) > track.track_width:
            valid_data = False

        if ntar_st.p.s > track.track_length/2.0+0.5:
            valid_data = False
        
        if abs(ntar_st.p.x_tran) > track.track_width:
            valid_data = False

        if abs(ntar_st.p.s - tar_st.p.s) > 1.0:
            valid_data = False

        return valid_data
    
    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return (self.samples[self.counter - 1], self.output_data[self.counter - 1])
        
    def get_datasets(self, filter = False):        
        not_done = True
        sample_idx = 0        
        ## filter 
        # if filter:
        #     # samples = [item for item in self.samples if sum((item[:,0] < 0.5) and item[:,0] > 0.5) > 2]
        #     samples = [item for item in self.samples if sum((0.5 > item[:, 0]) & (item[:, 0] > 0.0)) > 2]
        #     samp_len = len(samples)
        # else:
        

        if self.normalized_output is None:
            inputs= torch.stack(self.samples).to(torch.device("cuda"))  
            labels = torch.stack(self.output_data).to(torch.device("cuda"))
        else:
            inputs = self.normalized_sample.to(torch.device("cuda"))  
            labels = self.normalized_output.to(torch.device("cuda"))  
        # self.means_y = labels.mean(dim=0, keepdim=True)
        # self.stds_y = labels.std(dim=0, keepdim=True)
        # labels = (labels - self.means_y) / self.stds_y
        # perm = torch.randperm(len(inputs))
        # inputs = inputs[perm]
        # labels = labels[perm]
        samp_len = self.getNumSamples()            
        dataset =  torch.utils.data.TensorDataset(inputs,labels) 
        train_size = int(0.8 * samp_len)
        val_size = int(0.1 * samp_len)
        test_size = samp_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
        # return train_dataset, val_dataset, test_dataset, self.means_y, self.stds_y



    def plotStatistics(self):
        print("no plot statics for this dataset")
        return
        

