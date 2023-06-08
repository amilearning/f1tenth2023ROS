import numpy as np

class VAEGPPredictor():
    def __init__(self):
        
        self.GPmodel = None
        self.VAEModel = None

    def param_update(self):
        return 

    def vae_forward(self,ego_state,tar_state): 
        input_for_vae = self.VAEModel.state_preprocess(ego_state,tar_state)         
        theta = self.VAEModel(input_for_vae)
        return theta

    def gp_forward(self,theta,ego_state,tar_state):        
        input_for_gp = self.VAEModel.state_preprocess(theta,ego_state,tar_state)         
        output_from_gp = self.GPmodel(input_for_gp)
        return output_from_gp
    
    def input_predict(self,ego_state,tar_state):
        self.theta = self.vae_forward(ego_state,tar_state)
        predicted_target_u = self.gp_forward(self.theta,ego_state,tar_state)
        return predicted_target_u, self.theta 
    
    def model_update(self, load_from_file = False):
        if load_from_file: 
            print("load from saved model")
        else:
            print("update model with trained dada")
        return 

    def model_save(self):
        # save data for training 
        return

