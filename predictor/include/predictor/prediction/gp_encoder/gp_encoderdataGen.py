import torch
from barcgp.common.utils.scenario_utils import *
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *    
from barcgp.common.utils.scenario_utils import policy_generator, interp_state_with_vel

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
                        ego_st.p.x_tran,
                        ego_st.p.e_psi, 
                        ego_st.v.v_long,                       
                        tar_st.lookahead.curvature[2]])
    return input_data


class SampleGeneartorGPContEncoder(SampleGenerator):
    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):
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
        self.input_dim = 9
        self.time_horizon = 5
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.output_data = []
        self.info = []
        
        
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    # scenario_data: SimData = pickle.load(dbfile)
                    scenario_data: RealData = pickle.load(dbfile)
                    
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
                            dat = torch.zeros(self.time_horizon, self.input_dim)
                            
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
                                dat[i-t,:]=states_to_encoder_input_torch(tar_st, ego_st)
                                # torch.tensor([ tar_st.p.s - ego_st.p.s,
                                #                             ego_st.p.x_tran,
                                #                             ego_st.p.e_psi,
                                #                             ego_st.lookahead.curvature[0],                                                            
                                #                             tar_st.p.x_tran,
                                #                             tar_st.p.e_psi,
                                #                             tar_st.lookahead.curvature[0]])
                                    
                            
                            ### Add curvature[2] at the last dimension 
                            
                            next_tar_st = ntar_orin
                            real_dt = next_tar_st.t - tar_st.t 
                            valid_data = self.data_validation(ego_st,tar_st,next_tar_st,scenario_data.track)                        
                            if valid_data and real_dt > 0.05:
                                dt = 0.1                        
                                ntar_st = interp_state_with_vel(scenario_data.track, tar_st,next_tar_st,dt).copy()
                                # state_input = torch.tensor([tar_st.p.x_tran,
                                #                             tar_st.p.e_psi,                                                            
                                #                             tar_st.v.v_long,                                                           
                                #                             tar_st.v.v_tran,                                                           
                                #                             tar_st.lookahead.curvature[0],
                                #                             tar_st.lookahead.curvature[2]]).to(torch.device("cuda"))  
                                state_input = torch.tensor([   tar_st.p.x_tran,
                                                                tar_st.p.e_psi,                                                            
                                                                tar_st.v.v_long,                                          
                                                                tar_st.lookahead.curvature[0],                                                            
                                                                tar_st.lookahead.curvature[2]]).to(torch.device("cuda"))  
                                
                                gp_output = torch.tensor([next_tar_st.p.x_tran-tar_st.p.x_tran, next_tar_st.p.e_psi-tar_st.p.e_psi, next_tar_st.v.v_long-tar_st.v.v_long])
                                self.samples.append(dat)  
                                self.output_data.append(gp_output)    
                            
                        
                    
                    dbfile.close()
                
        print('Generated Dataset with', len(self.samples), 'samples!')
        
        # if randomize:
        #     random.shuffle(self.samples)
        
     
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
        inputs= torch.stack(self.samples).to(torch.device("cuda"))  
        labels = torch.stack(self.output_data).to(torch.device("cuda"))
        perm = torch.randperm(len(inputs))
        inputs = inputs[perm]
        labels = labels[perm]
        samp_len = self.getNumSamples()            
        dataset =  torch.utils.data.TensorDataset(inputs,labels) 
        train_size = int(0.8 * samp_len)
        val_size = int(0.1 * samp_len)
        test_size = samp_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        return train_dataset, val_dataset, test_dataset



    def plotStatistics(self):
        print("no plot statics for this dataset")
        return
        

