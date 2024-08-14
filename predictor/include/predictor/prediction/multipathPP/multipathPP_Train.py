#!/usr/bin/env python3

from predictor.common.utils.file_utils import *
import numpy as np
import torch
from datetime import datetime

from torch.utils.data import DataLoader, random_split


# from predictor.prediction.covGP.covGPNN_model import COVGPNN

from predictor.prediction.multipathPP.multipathPP_model import MULTIPATHPP

from predictor.prediction.multipathPP.multipathPP_dataGen import SampleGeneartorMultiPathPP


# Training
def multipathpp_train(dirs = None, val_dirs = None, real_data = False, args = None):
   
    if args is None:
        print("ARGS should be given!!")
        return 
    sampGen = SampleGeneartorMultiPathPP(dirs, args = args, randomize=True, real_data = real_data)
    # sampGen.plotStatistics()
    valid_args = args.copy()
  
  
    valGEn = SampleGeneartorMultiPathPP(val_dirs, load_normalization_constant = True, args = valid_args, randomize=False, real_data = True, tsne = False)
 
    
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

  
        
    
    multipathpp_predictor = MULTIPATHPP(args, sampGen, enable_GPU=True)
    
    # if args["train_nn"l] is False:
    # snapshot_name = 'traceGP_3000snapshot'
    # covgp_predictor.load_model(snapshot_name)
    # print(" model loaded")
    
    multipathpp_predictor.train(sampGen, valGEn, args = args)
    multipathpp_predictor.set_evaluation_mode()    

    create_dir(path=model_dir)    
    model_name = 'multipathpp'
    multipathpp_predictor.save_model(model_name)
    # covgp_predictor.load_model(gp_name)
    # covgp_predictor.evaluate()

