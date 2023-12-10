#!/usr/bin/env python3

from barcgp.common.utils.file_utils import *
import numpy as np
import torch
import gpytorch

from barcgp.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate

from barcgp.prediction.encoder.encoderModel import LSTMAutomodel
from barcgp.prediction.encoder.policyEncoder import PolicyEncoder
from torch.utils.data import DataLoader, random_split

from barcgp.prediction.thetaGP.ThetaGPdataGen import SampleGeneartorThetaGP
from barcgp.prediction.thetaGP.ThetaGPModel import ThetaGPApproximate
from barcgp.prediction.covGP.covGPNN_model import COVGPNN
from barcgp.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP


# Training
def covGPNN_train(dirs = None):
    
    sampGen = SampleGeneartorCOVGP(dirs, randomize=True)
    
    sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    args = {                    
            "batch_size": 512,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "input_dim": 9,
            "n_time_step": 10,
            "latent_dim": 8,
            "gp_output_dim": 4,
            "batch_size": 100,
            "inducing_points" : 300                
            }
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args["gp_output_dim"]) 
    covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
                     
    
    covgp_predictor.train(sampGen)
    covgp_predictor.set_evaluation_mode()
    trained_model = covgp_predictor.model, covgp_predictor.likelihood

    create_dir(path=model_dir)
    gp_name = 'covGP'
    covgp_predictor.save_model(gp_name)
    # covgp_predictor.load_model(gp_name)
    # covgp_predictor.evaluate()


# if __name__ == "__main__":
#     covGPNN_train()
