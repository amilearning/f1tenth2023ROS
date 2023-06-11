from barcgp.common.utils.scenario_utils import *

from barcgp.prediction.dyn_prediction_model import DynamicsModelForPredictor

class SampleGeneartorDynGP(SampleGenerator):
    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):



        # Input for DynGP ->         
        #               [vlong, vlat, wpsi, u_a, u_steer]       
        # Output from DynGP
        #               [del_vlong, del_vtran, del_wpsi]
        
        self.input_dim = 5
        self.output_dim = 3
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.output_data = []
        self.info = []
        self.dyn_model = DynamicsModelForPredictor()
                
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    scenario_data: SimData = pickle.load(dbfile)
                    N = scenario_data.N 
                    if N > 3:                      
                        for i in range(N-1):
                            if scenario_data.ego_states[i] is not None:
                                ego_st = scenario_data.ego_states[i]                                
                                action = np.array([ego_st.u.u_a , ego_st.u.u_steer])                                 
                                ntar_st_dyn = self.dyn_model.DynamicsUpdate(ego_st.copy(),action.copy())
                                next_ego_st = scenario_data.ego_states[i + 1]
                                
                                del_v_long = (next_ego_st.v.v_long - ntar_st_dyn.v.v_long)
                                del_v_tran = (next_ego_st.v.v_tran - ntar_st_dyn.v.v_tran)
                                del_w_psi = (next_ego_st.w.w_psi - ntar_st_dyn.w.w_psi)
                                
                                dyngp_input = torch.tensor([ego_st.v.v_long, 
                                                            ego_st.v.v_tran,
                                                            ego_st.w.w_psi,
                                                            ego_st.u.u_a,
                                                            ego_st.u.u_steer]).to(torch.device("cuda"))  
                                dyngp_output = torch.tensor([del_v_long,
                                                            del_v_tran,
                                                            del_w_psi]).to(torch.device("cuda"))  
                                self.samples.append(dyngp_input)  
                                self.output_data.append(dyngp_output)  
                            
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

  

    # def DynamicsUpdate(self,tar_state,action):

    #     next_state = tar_state.copy()     
    #     x = np.array([tar_state.p.s,tar_state.p.x_tran,tar_state.p.e_psi,tar_state.v.v_long,tar_state.v.v_tran,tar_state.w.w_psi])
    #     curs = np.array(tar_state.lookahead.curvature[0])
    #     u = np.array([action[0],action[1]])
    #     # x: s(0), ey(1), epsi(2), vx(3), vy(4), wz(5) , u(0) = ax, u(1) = delta                
        
    #     # x[:,2] = wrap_to_pi(x[:,2])
    #     while x[2] > np.pi-0.01:
    #         x[2] -= 2.0 * np.pi
    #     while x[2] < -np.pi+0.01:
    #         x[2] += 2.0 * np.pi

    #     nx = x.copy()
    #     self.dt = 0.1                    
    #     self.Lr = 0.13
    #     self.Lf = 0.13
    #     self.m = 2.366
    #     self.L = self.Lr+self.Lf
    #     self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/np.pi
    #     self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/np.pi
    #     self.Izz = 0.018  # self.Lf*self.Lr*self.m
    #     self.g= 9.814195
    #     self.h = 0.15

    #     self.Bp = 1.0 # 1.0
    #     self.Cp = 1.25 # 1.25   
    #     self.wheel_friction = 0.8 # 0.8            
    #     self.Df = self.wheel_friction*self.m*self.g * self.Lr / (self.Lr + self.Lf)        
    #     self.Dr = self.wheel_friction*self.m*self.g * self.Lf / (self.Lr + self.Lf)       

     
    #     ################  Pejekap 
    #     clip_vx = np.max([x[3],1.0])
    #     alpha_f_p = u[1] - np.arctan2(x[4]+self.Lf*x[5], clip_vx)
        
    #     alpha_r_p = - np.arctan2(x[4]-self.Lr*x[5] , clip_vx)
        
    #     Fyf = self.Df*np.sin(self.Cp*np.arctan(self.Bp*alpha_f_p))
    #     Fyr = self.Dr*np.sin(self.Cp*np.arctan(self.Bp*alpha_r_p))

    #     axb = u[0]
    #     delta = u[1]

    #     s = x[0]
    #     ey = x[1]        
    #     epsi = x[2]         
    #     vx = x[3]
    #     vy = x[4]
    #     wz = x[5]        
        
    #     nx[0] = s + self.dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) )
    #     nx[1] = ey + self.dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    #     nx[2] = epsi + self.dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curs * ey) * curs )
    #     nx[3] = vx + self.dt * (axb - 1 / self.m * Fyf * np.sin(delta) + wz*vy)
    #     nx[3] = np.max([nx[3],0.0])
    #     nx[4] = vy + self.dt * (1 / self.m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
    #     nx[5] = wz + self.dt * (1 / self.Izz *(self.Lf * Fyf * np.cos(delta) - self.Lf * Fyr) )

    #     next_state.p.s = nx[0]
    #     next_state.p.x_tran = nx[1]
    #     next_state.p.e_psi = nx[2]
    #     next_state.v.v_long = nx[3]
    #     next_state.v.v_tran = nx[4]
    #     next_state.w.w_psi = nx[5]
    #     return next_state
        


    def plotStatistics(self):
        print("no plot statics for this dataset")
        return
        

     
    