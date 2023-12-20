from predictor.common.utils.file_utils import *
from predictor.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from predictor.prediction.thetaGP.ThetaGPTrain import thetagp_train
# from predictor.prediction.gp_berkely_train import gp_main
from predictor.prediction.covGP.covGPNN_Train import covGPNN_train
import os

def get_dir(policy_name_, train_dir_):
    policy_dir = os.path.join(train_dir_, policy_name_)
    scencurve_dir = os.path.join(policy_dir, 'curve')
    scenstraight_dir = os.path.join(policy_dir, 'straight')
    scenchicane_dir = os.path.join(policy_dir, 'chicane')
    dirs = [scencurve_dir, scenstraight_dir, scenchicane_dir]
    return dirs


# timid = os.path.join(real_dir, 'lowspeed_nonblocking')
# block = os.path.join(real_dir, 'lowspeed_blocking')



dirs = [real_dir]
# dirs = [timid]

def main():  
    # print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("GP Berkely train init")
    # gp_main(dirs)
    # print("GP Berkely train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("covGPNN_train train init")
    covGPNN_train(dirs, real_data = True)
    print("AutoEncoder train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

  


if __name__ == "__main__":
    main()





