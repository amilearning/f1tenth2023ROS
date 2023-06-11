#!/usr/bin/env python3

from barcgp.common.utils.file_utils import *
import numpy as np
import torch
import gpytorch
from barcgp.prediction.dynGP.dynGPdataGen import SampleGeneartorDynGP
from barcgp.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate
from barcgp.prediction.dynGP.dynGPmodel import DynGPApproximate


# a_policy_name = 'aggressive_blocking'
# a_policy_dir = os.path.join(train_dir, a_policy_name)
# a_scencurve_dir = os.path.join(a_policy_dir, 'curve')
# a_scenstraight_dir = os.path.join(a_policy_dir, 'straight')
# a_scenchicane_dir = os.path.join(a_policy_dir, 'chicane')

# t_policy_name = 'timid'
# t_policy_dir = os.path.join(train_dir, t_policy_name)
# t_scencurve_dir = os.path.join(t_policy_dir, 'curve')
# t_scenstraight_dir = os.path.join(t_policy_dir, 'straight')
# t_scenchicane_dir = os.path.join(t_policy_dir, 'chicane')

# Training
def dyngp_train(dirs):
    # dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir, t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]    
    # dirs = [a_scencurve_dir]    
    
    
    sampGen = SampleGeneartorDynGP(dirs, randomize=True)
    
    sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=sampGen.output_dim) 
    Inputgp_predictor = DynGPApproximate(sampGen, IndependentMultitaskGPModelApproximate, likelihood,
                                            input_size=sampGen.input_dim, output_size=sampGen.output_dim, inducing_points=200, enable_GPU=True)
    
    Inputgp_predictor.train()
    Inputgp_predictor.set_evaluation_mode()
    trained_model = Inputgp_predictor.model, Inputgp_predictor.likelihood

    create_dir(path=model_dir)
    gp_name = 'dynGP'
    Inputgp_predictor.save_model(gp_name)
    #Inputgp_predictor.evaluate()


# if __name__ == "__main__":
#     main()
