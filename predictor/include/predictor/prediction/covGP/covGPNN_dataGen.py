import torch
from predictor.common.utils.scenario_utils import *
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.h2h_configs import *    
from predictor.common.utils.scenario_utils import policy_generator, wrap_del_s
from predictor.common.utils.file_utils import *
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
                        ]).clone()
    
    return input_data



class SampleGeneartorCOVGP(SampleGenerator):
    def __init__(self, abs_path,args = None, real_data = False, load_normalization_constant = False, pre_load_data_name = None, randomize=False, elect_function=None, init_all=True, tsne = False):
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
        self.input_dim = args["input_dim"]        
        self.time_horizon = args["n_time_step"]
        self.add_noise_data = args["add_noise_data"]
        model_name = args["model_name"]
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.output_data = []
        self.info = []
        self.debug_t = []
        self.means_y = None
        self.stds_y = None
        invalid_data_count = 0
        # pre_load_data_name = "preload_data"
        # if not dest_path.exists()        
        if pre_load_data_name is not None:       
            pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')              
            pre_data_dir =os.path.join(pre_data_dir,pre_load_data_name+'.pkl')
            self.load_pre_data(pre_data_dir)            
        else:            
            pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')      
            create_dir(pre_data_dir)        
            pre_data_dir =os.path.join(pre_data_dir,"preload_data.pkl")        
            if model_name == 'naiveGP':
                self.gen_samples(real_data = real_data, tsne = tsne, pre_data_dir = pre_data_dir)
            else:
                self.gen_samples_with_buffer(real_data = real_data, tsne = tsne, pre_data_dir = pre_data_dir)

        
        
        if len(self.samples) < 1 or self.samples is None:
            return 
        # self.plotStatistics()
        if args['model_name'] is not None:
            self.input_output_normalizing(name = args['model_name'], load_constants=load_normalization_constant)
        print('Generated Dataset with', len(self.samples), 'samples!')
       
        # if randomize:
        #     random.shuffle(self.samples)
    

    def gen_samples_with_buffer(self, real_data = False, tsne = False, pre_data_dir = None):
            invalid_data_count = 0
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        if real_data:
                            scenario_data: SimData = pickle.load(dbfile)                                
                            track_ = scenario_data.track
                        else:
                            scenario_data: RealData = pickle.load(dbfile)                                                            
                            track_ = scenario_data.scenario_def.track

                        # scenario_data: RealData = pickle.load(dbfile)                        
                        N = scenario_data.N              
                        ######################## random Policy ############################
                        if real_data:
                            policy_name = ab_p.split('/')[-1]
                        else:
                            policy_name = ab_p.split('/')[-2]
                        policy_gen = False
                        if policy_name == 'wall' :
                            policy_gen = True
                            # tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.scenario_def.track)                    
                            # tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.track)                    
                            tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=track_)                                        
                        ###################################################################
                        if N > self.time_horizon+1:
                            for t in range(N-1-self.time_horizon*2):                            
                                # define empty torch with proper size 
                                dat = torch.zeros(self.input_dim, 2*self.time_horizon)
                                # fdat = torch.zeros(self.input_dim, self.time_horizon)
                                for i in range(t,t+self.time_horizon):                                
                                    ego_st = scenario_data.ego_states[i]
                                    tar_st = scenario_data.tar_states[i]
                                    ntar_orin = scenario_data.tar_states[i+1]
                                    real_dt = ntar_orin.t - tar_st.t 
                                
                                    if policy_gen and i < t+self.time_horizon-3:
                                        scenario_data.tar_states[i+1] = policy_generator(tar_dynamics_simulator,scenario_data.tar_states[i])                  
                                        ntar_orin = scenario_data.tar_states[i+1]
                                    # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                    #                 tar_ey, tar_epsi, tar_cur]      
                                                               
                                    dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st)                                          
                                    dat[0,i-t] = wrap_del_s(tar_st.p.s,ego_st.p.s , track_)

                                                                        
                                next_tar_st = ntar_orin.copy()
                                real_dt = next_tar_st.t - tar_st.t 
                                valid_data = self.data_validation(dat[:,:self.time_horizon],tar_st,next_tar_st,track_)                                                        
                                if tsne:
                                    if tar_st.v.v_long < 0.05 or abs(ego_st.p.s - tar_st.p.s) > 1.3:
                                        valid_data = False

                                if valid_data and (real_dt > 0.05 and real_dt < 0.2):                              
                                    # delta_s = next_tar_st.p.s-tar_st.p.s
                                    delta_s = wrap_del_s(next_tar_st.p.s,tar_st.p.s, track_)                                    
                                    delta_xtran = next_tar_st.p.x_tran-tar_st.p.x_tran
                                    delta_epsi = next_tar_st.p.e_psi-tar_st.p.e_psi
                                    delta_vlong  = next_tar_st.v.v_long-tar_st.v.v_long
                                    

                                    self.debug_t.append(real_dt)
                                    
                                        
                                    for i in range(t+self.time_horizon, t+self.time_horizon*2):
                                        ego_st = scenario_data.ego_states[i]
                                        tar_st = scenario_data.tar_states[i]
                                        ntar_orin = scenario_data.tar_states[i+1]
                                        
                                    
                                        # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                        #                 tar_ey, tar_epsi, tar_cur]                                 
                                        dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st)                                        
                                        dat[0,i-t] = wrap_del_s(tar_st.p.s,ego_st.p.s , track_)

                                    gp_output = torch.tensor([delta_s, delta_xtran, delta_epsi, delta_vlong ])                                                                                                                               

                                    if self.add_noise_data:
                                        # if np.random.randint(10) < 8:
                                        noisy_dat, noisy_output = self.add_noise(dat, gp_output)
                                        self.output_data.append(noisy_output.clone())    
                                        self.samples.append(noisy_dat.clone())  
                                    self.output_data.append(gp_output.clone())    
                                    self.samples.append(dat.clone())  
                                else:
                                    invalid_data_count+=1
                                ### Add curvature[2] at the last dimension 
                    
                        dbfile.close()
            self.save_pre_data(pre_data_dir)


    def gen_samples(self, real_data = False, tsne = False, pre_data_dir = None):
            invalid_data_count = 0
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        if real_data:
                            scenario_data: SimData = pickle.load(dbfile)                                
                            track_ = scenario_data.track
                        else:
                            scenario_data: RealData = pickle.load(dbfile)                                                            
                            track_ = scenario_data.scenario_def.track
                        # scenario_data: RealData = pickle.load(dbfile)                        
                        N = scenario_data.N              
                        ######################## random Policy ############################
                        if real_data:
                            policy_name = ab_p.split('/')[-1]
                        else:
                            policy_name = ab_p.split('/')[-2]
                                                            
                        ###################################################################
                        if N > self.time_horizon+1:
                            
                            for t in range(N-1-self.time_horizon*2):                            
                                # define empty torch with proper size 
                                dat = torch.zeros(self.input_dim, 1)
                                # fdat = torch.zeros(self.input_dim, self.time_horizon)                                                                
                                ego_st = scenario_data.ego_states[t]
                                tar_st = scenario_data.tar_states[t]
                                ntar_orin = scenario_data.tar_states[t+1]
                                real_dt = ntar_orin.t - tar_st.t 
                                dat=states_to_encoder_input_torch(tar_st, ego_st).unsqueeze(dim=1)      
                                
                                next_tar_st = ntar_orin.copy()
                                real_dt = next_tar_st.t - tar_st.t 
                                valid_data = self.data_validation(dat,tar_st,next_tar_st,track_)                                                        
                                if tsne:
                                    if tar_st.v.v_long < 0.05 or abs(ego_st.p.s - tar_st.p.s) > 1.3:
                                        valid_data = False

                                if valid_data and (real_dt > 0.05 and real_dt < 0.2):                              
                                    delta_s = wrap_del_s(next_tar_st.p.s, tar_st.p.s, track_)                                    
                                    delta_xtran = next_tar_st.p.x_tran-tar_st.p.x_tran
                                    delta_epsi = next_tar_st.p.e_psi-tar_st.p.e_psi
                                    delta_vlong  = next_tar_st.v.v_long-tar_st.v.v_long                                    
                                    self.debug_t.append(real_dt)
                                    gp_output = torch.tensor([delta_s, delta_xtran, delta_epsi, delta_vlong ])                                                                                                                               

                                    if self.add_noise_data:
                                        # if np.random.randint(10) < 8:
                                        for i in range(4):
                                            noisy_dat, noisy_output = self.add_noise(dat, gp_output)
                                            self.output_data.append(noisy_output.clone())    
                                            self.samples.append(noisy_dat.clone())  
                                    self.output_data.append(gp_output.clone())    
                                    self.samples.append(dat.clone())  
                                else:
                                    invalid_data_count+=1
                                ### Add curvature[2] at the last dimension 
                        
                        dbfile.close()
            self.save_pre_data(pre_data_dir)


    def add_noise(self, input, output, input_noise_level = None):
        # measurement noise level
                    # delta_s, tar_xtran, tar_epsi,      tar_vx, 
        if input_noise_level is None:
            input_noise_level = torch.tensor([0.3,       0.3,     10*np.pi/180,  0.2])
        output_noise_level = torch.tensor([0.3,       0.3,     10*np.pi/180,  0.2])
        
        noisy_input = input.clone()
        noisy_output = output.clone()
        for i in range(4):
            if noisy_input.shape[1] > 1:            
                    noisy_input[i,:8] += np.random.uniform(-input_noise_level[i], input_noise_level[i], (8))
            else:
                noisy_input[i] += np.random.uniform(-input_noise_level[i], input_noise_level[i], 1)
        for i in range(4):
            noisy_output[i] += np.random.uniform(-output_noise_level[i], output_noise_level[i], 1)

        return noisy_input, noisy_output

    def get_eval_data(self,abs_path,pred_horizon = 10, real_data = True, noise = False):
        # pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')      
        # create_dir(pre_data_dir)        
        # pre_data_dir =os.path.join(pre_data_dir,"eval_preload_data.pkl")                        
        input_buffer_list = [] 
        ego_state_list = []
        tar_state_list = []
        gt_tar_state_list = []
        for ab_p in abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    if real_data:
                        scenario_data: SimData = pickle.load(dbfile)                                
                        track_ = scenario_data.track
                    else:
                        scenario_data: RealData = pickle.load(dbfile)                                                            
                        track_ = scenario_data.scenario_def.track

                    
                    # scenario_data: RealData = pickle.load(dbfile)                        
                    N = scenario_data.N   
                    if N > self.time_horizon+5:
                        for t in range(N-1-self.time_horizon - pred_horizon):                            
                            # define empty torch with proper size 
                            encoder_input = torch.zeros(self.input_dim, self.time_horizon)                            
                            for i in range(t,t+self.time_horizon):                                
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i]
                                encoder_input[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st)
                                encoder_input[0,i-t] = wrap_del_s(tar_st.p.s,ego_st.p.s , track_)
                                
                            ego_state =  scenario_data.ego_states[i]
                            tar_state =  scenario_data.tar_states[i]                            
                            tar_pred = VehiclePrediction()
                            tar_pred.s = array.array('d')
                            tar_pred.x_tran = array.array('d')
                            tar_pred.e_psi = array.array('d')
                            tar_pred.v_long = array.array('d')
                            for i in range(t+self.time_horizon-1,t+self.time_horizon+pred_horizon):
                                tar_pred.t = ego_state.t
                                tar_pred.s.append(scenario_data.tar_states[i].p.s)
                                tar_pred.x_tran.append(scenario_data.tar_states[i].p.x_tran)
                                tar_pred.e_psi.append(scenario_data.tar_states[i].p.e_psi)
                                tar_pred.v_long.append(scenario_data.tar_states[i].v.v_long)
                            
                            if noise:                                
                                input_noise_level = torch.tensor([0.1,       0.1,     10*np.pi/180,  0.1])
                                noisy_encoder_input , _ = self.add_noise(encoder_input,encoder_input, input_noise_level = input_noise_level)
                                
                                tar_state.p.x_tran = noisy_encoder_input[1,-1]
                                tar_state.p.e_psi = noisy_encoder_input[2,-1]
                                tar_state.v.v_long = noisy_encoder_input[3,-1]

                                ego_state.p.x_tran = noisy_encoder_input[6,-1]
                                ego_state.p.e_psi = noisy_encoder_input[7,-1]
                                ego_state.v.v_long = noisy_encoder_input[8,-1]
                                
                                

                            input_buffer_list.append(encoder_input)
                            ego_state_list.append(ego_state)
                            tar_state_list.append(tar_state)
                            gt_tar_state_list.append(tar_pred)

        return input_buffer_list, ego_state_list, tar_state_list, gt_tar_state_list

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


    def load_normalizing_consant(self, tensor_sample, tensor_output, name ='normalizing'):        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        means_x = model['mean_sample']
        means_y = model['mean_output']
        stds_x = model['std_sample']
        stds_y = model['std_output']   
        if len(tensor_sample.shape) ==2 :
            self.normalized_sample = (tensor_sample - means_x.repeat(tensor_sample.shape[0],1))/stds_x         
        elif len(tensor_sample.shape) ==3:
            self.normalized_sample = (tensor_sample - means_x.repeat(tensor_sample.shape[0],1,1))/stds_x      

        if len(tensor_output.shape) ==2 :
            self.normalized_output = (tensor_output - means_y.repeat(tensor_output.shape[0],1))/stds_y         
        elif len(tensor_output.shape) ==3:
            self.normalized_output = (tensor_output - means_y.repeat(tensor_output.shape[0],1,1))/stds_y      

        # self.independent = model['independent'] TODO uncomment        
        print('Successfully loaded normalizing constants', name)


    def input_output_normalizing(self,name = 'normalizing', load_constants = False):
        

        # x_h_tensor = torch.stack([sample[0] for sample in self.samples])
        # x_f_tensor = torch.stack([sample[1] for sample in self.samples])
        x_tensor = torch.stack( self.samples)

        tensor_output = torch.stack(self.output_data)
        
        if load_constants:
            self.load_normalizing_consant(x_tensor, tensor_output, name=name)
        else:
            self.normalized_sample, mean_sample, std_sample= self.normalize(x_tensor)
            self.normalized_output, mean_output, std_output = self.normalize(tensor_output)        
            model_to_save = dict()
            model_to_save['mean_sample'] = mean_sample
            model_to_save['std_sample'] = std_sample
            model_to_save['mean_output'] = mean_output
            model_to_save['std_output'] = std_output
            pickle_write(model_to_save, os.path.join(model_dir, name + '_normconstant.pkl'))
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
        delta_state = [nstate.p.s - kinmatic_nstate.p.s, nstate.p.x_tran - kinmatic_nstate.p.x_tran, nstate.p.e_psi - kinmatic_nstate.p.e_psi, nstate.v.v_long - state.v.v_long]
        return delta_state
        
        
     
    def data_validation(self,data : torch.tensor ,tar_st: VehicleState,ntar_st: VehicleState,track : RadiusArclengthTrack):
        valid_data = True

        delta_s = data[0,:]
        tar_ey = data[1,:]
        tar_epsi = data[2,:]
        tar_vx = data[3,:]
        k1 = data[4,:]
        k2 = data[5,:]
        ego_ey = data[6,:]
        ego_epsi = data[7,:]
        ego_vx = data[8,:]

        


        # # if ego_st.p.s > track.track_length/2.0+0.5 or tar_st.p.s > track.track_length/2.0+0.5:
        # #     valid_data = False
        
        # if abs(ego_st.p.x_tran) > track.track_width or abs(tar_st.p.x_tran) > track.track_width:
        #     valid_data = False

        # if ntar_st.p.s > track.track_length/2.0+0.5:
        #     valid_data = False
       

        # if abs(tar_ey).max() > track.track_width/2 or abs(ego_ey).max() > track.track_width/2:
        #     valid_data = False
        
        # if tar_vx.min() < 0.1:
        #     valid_data = False
            
        

        # if abs(ntar_st.p.x_tran) > track.track_width/2:
        #     valid_data = False
        del_s = wrap_del_s(ntar_st.p.s, tar_st.p.s, track)        
        if del_s is None:
            print("NA")
        if abs(del_s) > 1.5: #track.track_length/4:
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
        train_size = int(0.839 * samp_len)
        val_size = int(0.16 * samp_len)
        test_size = samp_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset
        # return train_dataset, val_dataset, test_dataset, self.means_y, self.stds_y



    def plotStatistics(self):
        
        # x_label = torch.stack(self.samples).detach().cpu().numpy()
        import matplotlib.pyplot as plt
        y_label = torch.stack(self.output_data).detach().cpu().numpy()
        debug_t = np.array(self.debug_t)
        titles = ["del_s", "del_ey", "del_epsi", "del_vx", "dt"]
        fig, axs = plt.subplots(5, 1, figsize=(10, 10))

        # Plot each column in a separate subplot
        for i in range(4):
            axs[i].plot(y_label[:, i])
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Sample Number')
            axs[i].set_ylabel('Value')

        axs[-1].plot(debug_t)
        axs[-1].set_title(titles[-1])
        axs[-1].set_xlabel('Sample Number')
        axs[-1].set_ylabel('Value')


        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
        
        print(1)
        
        
        return
        

