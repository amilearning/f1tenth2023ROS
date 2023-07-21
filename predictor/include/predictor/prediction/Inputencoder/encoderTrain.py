#!/usr/bin/env python3

from predictor.common.utils.file_utils import *
import numpy as np
import torch
from predictor.common.utils.scenario_utils import SampleGeneartorEncoder, SampleGenerator, Sample
from predictor.prediction.encoder.encoderModel import LSTMAutomodel
from predictor.prediction.encoder.policyEncoder import PolicyEncoder
from torch.utils.data import DataLoader, random_split


# a_policy_name = 'aggressive_blocking'
# a_policy_dir = os.path.join(train_dir, a_policy_name)
# sample_dir = os.path.join(a_policy_dir, 'sample')
# a_scencurve_dir = os.path.join(a_policy_dir, 'curve')
# a_scenstraight_dir = os.path.join(a_policy_dir, 'straight')
# a_scenchicane_dir = os.path.join(a_policy_dir, 'chicane')

# t_policy_name = 'timid'
# t_policy_dir = os.path.join(train_dir, t_policy_name)
# t_scencurve_dir = os.path.join(t_policy_dir, 'curve')
# t_scenstraight_dir = os.path.join(t_policy_dir, 'straight')
# t_scenchicane_dir = os.path.join(t_policy_dir, 'chicane')

# Training
def encoder_train(dirs):
    # dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir, t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]    
    # dirs = [sample_dir]    
    
    
    sampGen = SampleGeneartorEncoder(dirs, randomize=True)
    
  
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()
    args_ =  {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 11,
                "hidden_size": 8,
                "latent_size": 5,
                "learning_rate": 0.0001,
                "max_iter": 60000,
                "seq_len" :5
            }
    batch_size = args_["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    
    policy_encoder = PolicyEncoder(args= args_)
    
    policy_encoder.set_train_loader(train_loader)
    policy_encoder.set_test_loader(test_loader)

    policy_encoder.train(args= args_)
    
    create_dir(path=model_dir)
    policy_encoder.model_save()
    policy_encoder.tsne_evaluate()


# if __name__ == "__main__":
#     main()
