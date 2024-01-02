from predictor.common.utils.scenario_utils import *
from predictor.prediction.encoder.policyEncoder import PolicyEncoder
from predictor.simulation.dynamics_simulator import DynamicsSimulator
from predictor.h2h_configs import *    
from predictor.common.utils.scenario_utils import policy_generator

# def states_to_encoder_input_torch(tar_st,ego_st):
#     input_data=torch.tensor([ tar_st.p.s - ego_st.p.s,                        
#                         tar_st.p.x_tran,
#                         tar_st.p.e_psi,
#                         tar_st.v.v_long,
#                         tar_st.lookahead.curvature[0],
#                         ego_st.p.x_tran,
#                         ego_st.p.e_psi, 
#                         ego_st.v.v_long,                       
#                         ego_st.lookahead.curvature[0]])
#     return input_data


class SampleGeneartorEncoder(SampleGenerator):
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
        self.info = []
        
        
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    scenario_data: SimData = pickle.load(dbfile)
                    N = scenario_data.N                       
                    ######################## random Policy ############################
                    policy_name = ab_p.split('/')[-2]
                    policy_gen = False
                    if policy_name == 'wall':
                        policy_gen = True
                        tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=scenario_data.scenario_def.track)                    
                    ###################################################################
                    if N > self.time_horizon+5:
                        for t in range(N-1-self.time_horizon):                            
                            # define empty torch with proper size 
                            dat = torch.zeros(self.time_horizon, self.input_dim)
                            
                            for i in range(t,t+self.time_horizon):                                
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i]
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
                                    
                                
                            self.samples.append(dat)      
                            
                        
                    
                    dbfile.close()
                
        print('Generated Dataset with', len(self.samples), 'samples!')
        
        # if randomize:
        #     random.shuffle(self.samples)
        
     
    
    def get_datasets(self):        
        not_done = True
        sample_idx = 0
        samples= torch.stack(self.samples).to(torch.device("cuda"))  
        samp_len = self.getNumSamples()            
        train_size = int(0.8 * samp_len)
        val_size = int(0.1 * samp_len)
        test_size = samp_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(samples, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset



   

