import torch
from predictor.common.utils.scenario_utils import *
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.h2h_configs import *    
from predictor.common.utils.scenario_utils import policy_generator, wrap_del_s
from predictor.common.utils.file_utils import *
def states_to_encoder_input_torch(tar_st,ego_st, track:RadiusArclengthTrack):
    tar_s = tar_st.p.s
    tar_s = wrap_s_np(tar_s,track.track_length)

    ego_s = ego_st.p.s
    ego_s = wrap_s_np(ego_s,track.track_length)
    ######### doubled track ##########
    # if tar_s > track.track_length/2.0:
    #     tar_s -=  track.track_length/2.0
    # if ego_s > track.track_length/2.0:
    #     ego_s -=  track.track_length/2.0

    # delta_s = tar_s - ego_s
    delta_s = wrap_del_s(tar_s, ego_s, track)
    
    if len(np.array(delta_s).shape) == 1 :
        delta_s = delta_s[0]
    if tar_st.lookahead.curvature[0] is None: 
        print(1)
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
        self.args = args
        self.input_dim = args["input_dim"]        
        self.time_horizon = args["n_time_step"]
        self.add_noise_data = args["add_noise_data"]
        self.add_aug_data = args["add_aug_data"]

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
        
        if args['eval'] is False:  
            if pre_load_data_name is not None:       
                pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')              
                pre_data_dir =os.path.join(pre_data_dir,pre_load_data_name+'.pkl')
                self.load_pre_data(pre_data_dir)            
            else:            
                pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')      
                create_dir(pre_data_dir)        
                pre_data_dir =os.path.join(pre_data_dir,"preload_data.pkl")        
                
                self.gen_samples_with_buffer(real_data = real_data, tsne = tsne, pre_data_dir = pre_data_dir)

            if len(self.samples) < 1 or self.samples is None:
                return 
        # self.plotStatistics()
            if randomize:
                self.samples, self.output_data = self.shuffle_in_out_data(self.samples, self.output_data)
            if args['model_name'] is not None:
                self.input_output_normalizing(name = args['model_name'], load_constants=load_normalization_constant)
            print('Generated Dataset with', len(self.samples), 'samples!')
       
        # if randomize:
        #     random.shuffle(self.samples)
    
    def shuffle_in_out_data(self,input, output):
        # Pair up the elements and shuffle
        combined = list(zip(input, output))
        random.shuffle(combined)

        # Unzip them back into two lists
        shuffled_list1, shuffled_list2 = zip(*combined)

        # Convert tuples back to lists (if needed)
        shuffled_input = list(shuffled_list1)
        shuffled_output = list(shuffled_list2)
        return shuffled_input, shuffled_output


    def gen_samples_with_buffer(self, real_data = False, tsne = False, pre_data_dir = None):
            invalid_data_count = 0
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        if real_data:
                            scenario_data: RealData = pickle.load(dbfile)                                
                            track_ = scenario_data.track
                        else:
                            scenario_data: SimData = pickle.load(dbfile)                                                            
                            track_ = scenario_data.scenario_def.track
                        # scenario_data: RealData = pickle.load(dbfile)                        
                        N = scenario_data.N              
                        ######################## random Policy ############################
                        if real_data:
                            policy_name = ab_p.split('/')[-1]
               
                        tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=track_)                                                      
                        ###################################################################
                        if N > self.time_horizon+1:
                            for t in range(N-1-self.time_horizon*2):  
                                #                if self.add_noise:
                                # tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.track)                    
                                # scenario_data.tar_states[i+1] = policy_generator(tar_dynamics_simulator,scenario_data.tar_states[i])                  
                                # ntar_orin = scenario_data.tar_states[i+1]
                                def get_x_y_data_from_index(t, scenario_data, tsne = False):                                                 
                                    # define empty torch with proper size                                     
                                    dat = torch.zeros(self.input_dim, 2*self.time_horizon)                                    
                                    for i in range(t,t+self.time_horizon):                                
                                        ego_st = scenario_data.ego_states[i]
                                        tar_st = scenario_data.tar_states[i]
                                        ntar_orin = scenario_data.tar_states[i+1]
                                        real_dt = ntar_orin.t - tar_st.t 
                                    # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                    #                 tar_ey, tar_epsi, tar_cur]      
                                                            
                                        dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track_)                                          
                                    # dat[0,i-t] = wrap_del_s(tar_st.p.s,ego_st.p.s , track_)
                            
                                    next_tar_st = scenario_data.tar_states[t+self.time_horizon].copy()
                                    tar_st = scenario_data.tar_states[t+self.time_horizon-1].copy()
                                    
                                    valid_data = self.data_validation(dat[:,:self.time_horizon],tar_st,next_tar_st,track_)                                                        
                                    if tsne:
                                        del_s_tmp = wrap_del_s(tar_st.p.s, ego_st.p.s,track_)
                                        if tar_st.v.v_long < 0.05 or abs(del_s_tmp) > 1.0:                                            
                                        # if tar_st.v.v_long > 10.05 or del_s_tmp > 50.0:
                                            valid_data = False
                                        
                                    if valid_data:                              
                                        # delta_s = next_tar_st.p.s-tar_st.p.s
                                        delta_s = wrap_del_s(next_tar_st.p.s,tar_st.p.s, track_)                                    
                                        delta_xtran = next_tar_st.p.x_tran-tar_st.p.x_tran
                                        delta_epsi = next_tar_st.p.e_psi-tar_st.p.e_psi
                                        delta_vlong  = next_tar_st.v.v_long-tar_st.v.v_long
                                        
                                        self.debug_t.append(real_dt)
                                        gp_output = torch.tensor([delta_s, delta_xtran, delta_epsi, delta_vlong ]).clone()                                                                                                                               
                                        
                                        
                                        for i in range(t+self.time_horizon, t+self.time_horizon*2):
                                            ego_st = scenario_data.ego_states[i]
                                            tar_st = scenario_data.tar_states[i]
                                            ntar_orin = scenario_data.tar_states[i+1]
                                        
                                            # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                            #                 tar_ey, tar_epsi, tar_cur]                                 
                                            dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track_)                                        
                                            
                                        return dat.clone(), gp_output.clone() 
                                    else:
                                        return None, None # invalid_data_count+=1
                                
                                        ############### return noisy trajecoties ##############                                    
                                

                                    
                                tmp_sample, tmp_output = get_x_y_data_from_index(t,scenario_data, tsne)
                                if tmp_sample is None or tmp_output is None:
                                    continue
                                self.output_data.append(tmp_output)
                                self.samples.append(tmp_sample)

                                
                                if self.add_noise_data:
                                    if t < scenario_data.N-self.time_horizon*3 and t%5 ==0:
                                        aug_gp_outputs, aug_samples = self.get_additional_samples(False, t,scenario_data, track_, tar_dynamics_simulator)
                                        
                                        self.output_data.extend(aug_gp_outputs)
                                        self.samples.extend(aug_samples)

                                if self.add_aug_data:
                                    if t < scenario_data.N-self.time_horizon*3 and t%5 ==0:
                                        aug_gp_outputs, aug_samples = self.get_additional_samples(self.add_aug_data, t,scenario_data, track_, tar_dynamics_simulator)
                                        self.output_data.extend(aug_gp_outputs)
                                        self.samples.extend(aug_samples)
                                    
                                ### Add curvature[2] at the last dimension 
                    
                        dbfile.close()
            
            
            if self.args['model_name'] == 'naiveGP':                
                new_samples = []
                for i in range(len(self.samples)):
                    new_samples.append(self.samples[i][:,self.time_horizon-1])
                self.samples = new_samples
                
            self.save_pre_data(pre_data_dir)
            # print( "invalid_data_count = " + str(invalid_data_count))
            
    def get_additional_samples(self,is_augment, t, scenario_data: RealData or SimData, track: RadiusArclengthTrack, simulator : DynamicsSimulator, sample_num = 10):
        gp_outputs, samples = self.extend_dat(is_augment, t,sample_num, self.time_horizon*3,scenario_data,track,simulator)
        # # if is_augment:
        # original_dat = self.get_data(t,self.time_horizon*2, scenario_data, track)
        # original_dat2 = self.get_data(t,self.time_horizon*3, scenario_data, track)
        # x2 = range(self.time_horizon*3)
        # for i in range(len(samples)):
        #     x = range(i,i+len(samples[i][1,:]))
        #     plt.plot(x,samples[i][1,:], color='grey', alpha=0.1)
        # x = range(self.time_horizon*2)
        # plt.plot(x,original_dat[1,:], color='red', alpha=0.1)
        # plt.plot(x2,original_dat2[1,:],'r')
        

        return gp_outputs, samples
      

    def extend_dat(self,is_augment, t, num_sample, time_length, scenario_data: RealData or SimData, track: RadiusArclengthTrack, simulator: DynamicsSimulator):
        dat = torch.zeros(self.input_dim, time_length)                                    
        if time_length < self.time_horizon*2+1:
            time_length = self.time_horizon*2
        roll_tar_st = scenario_data.tar_states[t].copy()  
        # for i in range(t,t+time_length):            
        gp_outputs = []      
        samples = []
                      
        for i in range(time_length):                                
            ego_st = scenario_data.ego_states[i+t].copy()
            if i <=self.time_horizon-1:
                roll_tar_st = scenario_data.tar_states[i+t].copy()  
            tar_st = roll_tar_st.copy()               

            dat[:,i]=states_to_encoder_input_torch(tar_st, ego_st, track)  

            if is_augment:                                
                roll_tar_st = scenario_data.tar_states[i+t+1].copy()  
                roll_tar_st.p.s += np.random.randn(1)*0.01
                roll_tar_st.p.x_tran += np.random.randn(1)*0.01
                roll_tar_st.p.e_psi += np.random.randn(1)*0.01
                roll_tar_st.v.v_long += np.random.randn(1)*0.01                
            else:
                tmp_tar_st = scenario_data.tar_states[i+t+1].copy()  
                roll_tar_st = self.gen_random_next_state(tar_st,simulator, next_tar_state = tmp_tar_st)           
                
                if i >= self.time_horizon-1:
                    del_s = wrap_del_s(roll_tar_st.p.s,tar_st.p.s, track)                 
                    del_xtran = roll_tar_st.p.x_tran - tar_st.p.x_tran
                    del_epsi = roll_tar_st.p.e_psi - tar_st.p.e_psi
                    del_vx = roll_tar_st.v.v_long - tar_st.v.v_long
                    gp_output = torch.tensor([del_s, del_xtran, del_epsi, del_vx ]).clone()   
                    gp_outputs.append(gp_output.clone())

        if is_augment:
            dat[:,0:self.time_horizon*2] = self.augment_left_with_polynoial(dat[:,0:self.time_horizon*2])
        else:
            dat[:,:self.time_horizon] = self.get_left_dat_dynamics(t+self.time_horizon-1, scenario_data, simulator, track) 

        for k in range(num_sample):
            sample = dat[:,k:k+self.time_horizon*2]
            samples.append(sample.clone())
        
        new_samples = []
        new_gp_outputs = []
        for i in range(len(samples)):
            if gp_outputs[i][0] <= 0.3:
                new_samples.append(samples[i])
                new_gp_outputs.append(gp_outputs[i])

        
        return new_gp_outputs[:num_sample], new_samples
    
    def get_left_dat_dynamics(self,t, scenario_data, simulator, track):
        dat = torch.zeros(self.input_dim, self.time_horizon)         
        roll_tar_st = scenario_data.tar_states[t].copy()                               
        for i in range(self.time_horizon):                                
            ego_st = scenario_data.ego_states[i+t]
            if i == 0:
                roll_tar_st = scenario_data.tar_states[i+t].copy()                                      
            tar_st = roll_tar_st   
            dat[:,i]=states_to_encoder_input_torch(tar_st, ego_st, track)  
            roll_tar_st = self.gen_random_next_state(tar_st,simulator, direction=1.0, accel_mag = 1.0)  

        return torch.flip(dat, dims=[1])
    
    def get_data(self,t, time_length, scenario_data: RealData or SimData, track: RadiusArclengthTrack):
        dat = torch.zeros(self.input_dim, time_length)                                    
        for i in range(t,t+time_length):                                
            ego_st = scenario_data.ego_states[i]
            tar_st = scenario_data.tar_states[i]            
            dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track)  
        return dat

    
    def gen_random_next_state(self,state: VehicleState, simulator: DynamicsSimulator,next_tar_state = None,  direction = 1, accel_mag= 0.0):        
    
        tmp_state = state.copy()
        if tmp_state.p.e_psi < 0:            
            tmp_state.u.u_steer = tmp_state.u.u_steer -(np.random.rand(1)*0.05)[0] * direction
        else:
            tmp_state.u.u_steer = tmp_state.u.u_steer + (np.random.rand(1)*0.05)[0] * direction
        
        # noisy_steer = np.clip(np.random.randn(1)*0.4,-0.43, 0.43)[0]
        # # noisy_steer = tmp_state.u.u_steer + steer_noise[0] * direction
        
        tmp_state.u.u_a = accel_mag
        if tmp_state.p.x_tran > simulator.model.track.track_width*0.5:
            tmp_state.u.u_steer = -1*abs(np.random.randn(1)[0]*0.5)                
            tmp_state.u.u_a = -accel_mag
        elif tmp_state.p.x_tran < -1*simulator.model.track.track_width*0.7:
            tmp_state.u.u_steer = abs(np.random.randn(1)[0]*0.5)
            tmp_state.u.u_a = -accel_mag
        if next_tar_state is not None:
            tmp_state.u.u_steer = next_tar_state.u.u_steer #+ np.random.randn(1)[0]*0.00001
        
        tmp_state.u.u_steer = np.clip(tmp_state.u.u_steer, -0.43, 0.43)
        simulator.step(tmp_state) 
        # noise in longitudinal direction
        s_noise = np.clip(np.random.randn(1)*0.08, -0.032, 0.032)[0]
        if next_tar_state is None:
            tmp_state.p.s = tmp_state.p.s + s_noise
        else:            
            tmp_state.p.s = next_tar_state.p.s + s_noise


        simulator.model.track.update_curvature(tmp_state)      
        
        return tmp_state 
    
    

    def augment_left_with_polynoial(self, data):
        # data -> b, d, t -> d= 9, t = 10
        # del_s, tar_s, tar_ey, tar_epsi, vx 
        out_data = data.clone()
        noise_levels = torch.tensor([10.0, 10.0, 10.0, 10.0])
        for idx in range(len(noise_levels)):            
            noise = self.generate_quintic_polynomial(-noise_levels[idx],noise_levels[idx])
            out_data[idx,:int(out_data.shape[1]/2)-1] += noise[:int(out_data.shape[1]/2)-1] * 1e-3
        return out_data 
            

    def generate_quintic_polynomial(self, y_min, y_max, max_attempts=1000):
        x_start, x_end = -1, 1
        for _ in range(max_attempts):
            # Generate random coefficients for a quintic polynomial
            coeffs = np.random.uniform(-3, 3, 6)
            coeffs[-1] = 0 # ensure it passes the origin
            poly = np.poly1d(coeffs)

            # Evaluate the polynomial over the specified x-axis range
            x = np.linspace(x_start, x_end, 21)
            y = poly(x)
            
            # Check if all y-values are within the specified limits
            if np.all((y >= y_min.cpu().numpy()) & (y <= y_max.cpu().numpy())):
                return y

        raise ValueError("Unable to find a suitable polynomial within the specified limits.")

    






    def add_noise(self, input, output, noise_level = None):
        # measurement noise level
                    # delta_s, tar_xtran, tar_epsi,      tar_vx, 
        if noise_level is None:
            noise_level = torch.tensor([10,       10,     10,  10.0])
        
        noisy_input = input.clone()
        noisy_output = output.clone()
        noisy_output[1:] = -2*noisy_output[1:]
        noisy_output[1] = -5*noisy_output[1] # +np.random.uniform(-0.1, 0.1, (1))
        # noisy_output[0] += np.random.uniform(-noise_level[0], noise_level[0], (1))+0.1
        
        input_del_s_noise = self.generate_quintic_polynomial(-noise_level[0], noise_level[0])
        # input_del_s_noise = np.clip(input_del_s_noise, -2.0, 0.3)
        input_del_xtran_noise = self.generate_quintic_polynomial(-noise_level[1], noise_level[1])
        # input_del_xtran_noise = np.clip(input_del_xtran_noise, -0.3, 0.3)
        input_del_epsi_noise = self.generate_quintic_polynomial(-noise_level[2], noise_level[2])
        # input_del_epsi_noise = np.clip(input_del_epsi_noise, -30, 30)
        input_del_epsi_noise = input_del_epsi_noise*3.14/180.0
        input_del_vx_noise = self.generate_quintic_polynomial(-noise_level[3], noise_level[3])
        # input_del_vx_noise = np.clip(input_del_vx_noise, -2, 2)
        input_del_vx_noise = input_del_vx_noise/10.0

        output_del_s_noise = input_del_s_noise[11] - input_del_s_noise[10]
        output_del_xtran_noise = input_del_xtran_noise[11] - input_del_xtran_noise[10]
        output_del_epsi_noise = input_del_epsi_noise[11] - input_del_epsi_noise[10]
        output_del_vx_noise = input_del_vx_noise[11] - input_del_vx_noise[10]

        if noisy_input.shape[1] > 1:       
            ## train data for SimTSGP        
            noisy_input[0,:] += input_del_s_noise[1:]
            noisy_input[1,:] += input_del_xtran_noise[1:]
            noisy_input[2,:] += input_del_epsi_noise[1:]
            noisy_input[3,:] += input_del_vx_noise[1:]
  

        return noisy_input, noisy_output

    def get_eval_data(self,abs_path,pred_horizon = 10, real_data = True, noise = False):
        # pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')      
        # create_dir(pre_data_dir)        
        # pre_data_dir =os.path.join(pre_data_dir,"eval_preload_data.pkl")                        
        input_buffer_list = [] 
        track_angle_list = []
        ego_state_list = []
        tar_state_list = []
        gt_tar_state_list = []
        ego_pred_list= []
        track_list = []
        invalid_count= 0
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
                    if N > self.time_horizon+1:
                        for t in range(N-1-self.time_horizon - self.time_horizon):                                                        
                        
                            # define empty torch with proper size 
                            encoder_input = torch.zeros(self.input_dim, self.time_horizon)                            
                            for i in range(t,t+self.time_horizon):                                
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i]
                                encoder_input[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track_)                                
                                
                            ego_state =  scenario_data.ego_states[t+self.time_horizon-1].copy()
                            tar_state =  scenario_data.tar_states[t+self.time_horizon-1].copy()   
                            
                            ntar_state = scenario_data.tar_states[t+self.time_horizon].copy()                            
                            tar_pred = VehiclePrediction()
                            tar_pred.s = array.array('d')
                            tar_pred.x_tran = array.array('d')
                            tar_pred.e_psi = array.array('d')
                            tar_pred.v_long = array.array('d')

                            ego_pred = VehiclePrediction()
                            ego_pred.s = array.array('d')
                            ego_pred.x_tran = array.array('d')
                            ego_pred.e_psi = array.array('d')
                            ego_pred.v_long = array.array('d')

                            valid_data = self.data_validation(encoder_input,tar_state,ntar_state,track_)                                                        
                            ########## check the data near interaction                         
                            # del_tar_ego_s = wrap_del_s(tar_st.p.s, ego_st.p.s, track_)
                            # if abs(del_tar_ego_s) > 1.5:
                            #     valid_data = False
                            ########## check the data near interaction 

                            if valid_data is False:
                                invalid_count +=1
                                continue
                            
                          
                            for pred_j in range(t+self.time_horizon-1,t+self.time_horizon+self.time_horizon-1):
                                
                                tar_pred.t = ego_state.t
                                tar_pred.s.append(scenario_data.tar_states[pred_j].p.s)
                                tar_pred.x_tran.append(scenario_data.tar_states[pred_j].p.x_tran)
                                tar_pred.e_psi.append(scenario_data.tar_states[pred_j].p.e_psi)
                                tar_pred.v_long.append(scenario_data.tar_states[pred_j].v.v_long)

                                ego_pred.t = ego_state.t
                                ego_pred.s.append(scenario_data.ego_states[pred_j].p.s)
                                ego_pred.x_tran.append(scenario_data.ego_states[pred_j].p.x_tran)
                                ego_pred.e_psi.append(scenario_data.ego_states[pred_j].p.e_psi)
                                ego_pred.v_long.append(scenario_data.ego_states[pred_j].v.v_long)
                            
                            # if self.add_noise_data or noise:                                
             
                            #     noisy_encoder_input , _ = self.add_noise(encoder_input,encoder_input)
                                
                            #     tar_state.p.x_tran = noisy_encoder_input[1,-1]
                            #     tar_state.p.e_psi = noisy_encoder_input[2,-1]
                            #     tar_state.v.v_long = noisy_encoder_input[3,-1]

                            #     ego_state.p.x_tran = noisy_encoder_input[6,-1]
                            #     ego_state.p.e_psi = noisy_encoder_input[7,-1]
                            #     ego_state.v.v_long = noisy_encoder_input[8,-1]
                          
                            # if tar_pred.s[-1] == tar_pred.s[-2] == tar_pred.s[-3]:
                            #     print(1)

                            input_buffer_list.append(encoder_input.clone())
                            ego_state_list.append(ego_state.copy())
                            tar_state_list.append(tar_state.copy())
                            gt_tar_state_list.append(tar_pred.copy())
                            ego_pred_list.append(ego_pred.copy())
                            track_list.append(track_)
                            encoder_input = None
                            ego_state = None
                            tar_state = None
                            ego_pred = None
                            
        print("invalid_count = " + str(invalid_count))
        print("valid eval data len = " + str(len(input_buffer_list)))
        return input_buffer_list, ego_state_list, tar_state_list, gt_tar_state_list, ego_pred_list, track_list

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
            new_data = (data - mean.repeat(data.shape[0],1))/(std         + 1e-11)
        elif len(data.shape) ==3:
            new_data = (data - mean.repeat(data.shape[0],1,1))/(std         + 1e-11)
        return new_data, mean, std


    def load_normalizing_consant(self, tensor_sample, tensor_output, name ='normalizing'):        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample']
        self.means_y = model['mean_output']
        self.stds_x = model['std_sample']
        self.stds_y = model['std_output']   

        if tensor_sample is not None:            
            if len(tensor_sample.shape) ==2 :
                self.normalized_sample = (tensor_sample - self.means_x.repeat(tensor_sample.shape[0],1))/(self.stds_x + 1e-11)         
            elif len(tensor_sample.shape) ==3:
                self.normalized_sample = (tensor_sample - self.means_x.repeat(tensor_sample.shape[0],1,1))/(self.stds_x + 1e-11)
        if tensor_output is not None:
            if len(tensor_output.shape) ==2 :
                self.normalized_output = (tensor_output - self.means_y.repeat(tensor_output.shape[0],1))/(self.stds_y + 1e-11)        
            elif len(tensor_output.shape) ==3:
                self.normalized_output = (tensor_output - self.means_y.repeat(tensor_output.shape[0],1,1))/(self.stds_y+ 1e-11)      

        # self.independent = model['independent'] TODO uncomment        
        print('Successfully loaded normalizing constants', name)


    def input_output_normalizing(self,name = 'normalizing', load_constants = False):
        

        # x_h_tensor = torch.stack([sample[0] for sample in self.samples])
        # x_f_tensor = torch.stack([sample[1] for sample in self.samples])
        x_tensor = torch.stack(self.samples)

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

        # delta_s = data[0,:]
        # tar_ey = data[1,:]
        # tar_epsi = data[2,:]
        # tar_vx = data[3,:]
        # k1 = data[4,:]
        # k2 = data[5,:]
        # ego_ey = data[6,:]
        # ego_epsi = data[7,:]
        # ego_vx = data[8,:]

        


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
        # if del_s < 0.05: 
        #     valid_data = False
        
        # del_epsi = ntar_st.p.e_psi-  tar_st.p.e_psi
        # if abs(del_epsi) > 0.2: 
        #     valid_data = False

        # del_x_tran = ntar_st.p.x_tran-  tar_st.p.x_tran
        # if abs(del_x_tran) > 0.5: 
        #     valid_data = False

        del_vlong = ntar_st.v.v_long-  tar_st.v.v_long
        # if abs(del_vlong) > 0.5: 
        #     valid_data = False

        real_dt = ntar_st.t - tar_st.t 
        # if (real_dt < 0.05 or real_dt > 0.2):            
        #     valid_data = False

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
        train_size = int(1.0 * samp_len)
        val_size = int(0.01 * samp_len)
        test_size = samp_len - train_size - val_size
        
        # train_indices = list(range(0, train_size))
        # valid_indices = list(range(train_size, train_size+val_size))
        # test_indices = list(range(train_size+val_size, samp_len))

        train_dataset = dataset
        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        # val_dataset = train_dataset

        return train_dataset, dataset, dataset
        # return train_dataset, val_dataset, test_dataset, self.means_y, self.stds_y



    def plotStatistics(self):
        
        # x_label = torch.stack(self.samples).detach().cpu().numpy()
        import matplotlib.pyplot as plt
        # self.normalized_sample
        # self.normalized_output
        # self.output_data
        # y_label = torch.stack(self.output_data).detach().cpu().numpy()
        # self.normalized_sample[]
        x_label=torch.stack(self.samples).detach().cpu().numpy()        
        y_label=torch.stack(self.output_data).detach().cpu().numpy()
        # y_label = self.normalized_output.detach().cpu().numpy()
        debug_t = np.array(self.debug_t)
        titles = ["del_s", "del_ey", "del_epsi", "del_vx", "dt"]
        fig, axs = plt.subplots(5, 1, figsize=(10, 10))
        n_sample_to_plot = 1000
        selected_indices = list(range(y_label.shape[0]))
        if y_label.shape[0] > n_sample_to_plot:            
            selected_indices = random.sample(selected_indices, n_sample_to_plot)
        # Plot each column in a separate subplot
        for i in range(4):
            axs[i].plot(y_label[selected_indices,i])
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
        
        
        
        
        return
        

