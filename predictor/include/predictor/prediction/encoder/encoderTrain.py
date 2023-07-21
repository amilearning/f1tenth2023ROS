#!/usr/bin/env python3

from predictor.common.utils.file_utils import *
import numpy as np
import torch
from predictor.prediction.encoder.encoderdataGen import SampleGeneartorEncoder

from predictor.prediction.encoder.encoderModel import LSTMAutomodel
from predictor.prediction.encoder.policyEncoder import PolicyEncoder
from torch.utils.data import DataLoader, random_split


# Training
def encoder_train(dirs):

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
                "input_size": 9,
                "hidden_size": 8,
                "latent_size": 4,
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
