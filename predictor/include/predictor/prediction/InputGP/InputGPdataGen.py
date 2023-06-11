from barcgp.common.utils.scenario_utils import *
from barcgp.prediction.encoder.policyEncoder import PolicyEncoder

class SampleGeneartorInputGP(SampleGenerator):
    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):

        # Input for InputGP -> (t)         
        #               [(tar_s-ego_s),
        #                ego_ey, ego_epsi, ego_cur, ego_vlong, ego_vlat
        #                tar_ey, tar_epsi, tar_cur, ego_vlong, ego_vlat]         
        #                    +  
        #              output from Encoder         
        #                  
        # Output from InputGP
        #               [tar_accel(t+1), tar_delta(t+1)]
        self.encoder_model = PolicyEncoder()
        self.encoder_input_dim = self.encoder_model.input_dim
        self.encoder_time_horizon = self.encoder_model.seq_len
        self.encoder_output_dim = self.encoder_model.output_dim

        self.input_dim = self.encoder_output_dim + 13  + 4
        self.output_dim = 2
        
        self.encoder_model.model_load()

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
                    scenario_data: SimData = pickle.load(dbfile)
                    N = scenario_data.N                       
                    if N > self.encoder_time_horizon+5:
                        for t in range(N-1-self.encoder_time_horizon-1):                            
                            # define empty torch with proper size 
                            dat = torch.zeros(self.encoder_time_horizon, self.encoder_input_dim).to(torch.device("cuda"))  
                            for i in range(t,t+self.encoder_time_horizon):                                
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i] 
                                next_tar_st = scenario_data.tar_states[i+1]                               
                                # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
                                #                 tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta]                                 
                                dat[i-t,:]=torch.tensor([ tar_st.p.s - ego_st.p.s,
                                                            ego_st.p.x_tran,
                                                            ego_st.p.e_psi,
                                                            ego_st.lookahead.curvature[0],
                                                            ego_st.u.u_a,
                                                            ego_st.u.u_steer,
                                                            tar_st.p.x_tran,
                                                            tar_st.p.e_psi,
                                                            tar_st.lookahead.curvature[0],
                                                            tar_st.u.u_a,
                                                            tar_st.u.u_steer])
                            
                            theta = self.encoder_model.get_theta(dat,np = False).squeeze()
                            
                            state_input = torch.tensor([tar_st.p.s - ego_st.p.s,
                                                            tar_st.p.x_tran,
                                                            tar_st.p.e_psi,                                                            
                                                            tar_st.v.v_long,
                                                            tar_st.v.v_tran,
                                                            tar_st.w.w_psi,                                                                                                                        
                                                            ego_st.p.x_tran,
                                                            ego_st.p.e_psi,
                                                            ego_st.v.v_long,
                                                            ego_st.v.v_tran,
                                                            ego_st.w.w_psi,                                                            
                                                            tar_st.lookahead.curvature[0],
                                                            tar_st.lookahead.curvature[1],
                                                            tar_st.lookahead.curvature[2],
                                                            ego_st.lookahead.curvature[0],
                                                            ego_st.lookahead.curvature[1],
                                                            ego_st.lookahead.curvature[2]]).to(torch.device("cuda"))  
                            gp_input = torch.hstack([state_input, theta])  # 17 + theta_dim(5)   
                            # gp_input = state_input 
                            gp_output = torch.tensor([next_tar_st.u.u_a, next_tar_st.u.u_steer])
                            self.samples.append(gp_input)  
                            self.output_data.append(gp_output)    
                            
                    
                    dbfile.close()
                
        print('Generated Dataset with', len(self.samples), 'samples!')
        
        # if randomize:
        #     random.shuffle(self.samples)
    
    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return (self.samples[self.counter - 1], self.output_data[self.counter - 1])

    def get_datasets(self):        
        not_done = True
        sample_idx = 0
        samp_len = self.getNumSamples()           
        train_size = int(0.8 * self.getNumSamples())
        val_size = int(0.1 * self.getNumSamples())
        test_size = self.getNumSamples() - train_size - val_size

        inputs= torch.stack(self.samples).to(torch.device("cuda"))
        labels = torch.stack(self.output_data).to(torch.device("cuda"))
        perm = torch.randperm(len(inputs))
        inputs = inputs[perm]
        labels = labels[perm]
        dataset =  torch.utils.data.TensorDataset(inputs,labels)                
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset

  



    def plotStatistics(self):
        print("no plot statics for this dataset")
        return
        

     
    