from barcgp.common.utils.file_utils import *
from barcgp.prediction.dynGP.dynGP_train import dyngp_train
from barcgp.prediction.encoder.encoderTrain import encoder_train
from barcgp.prediction.Ikd.IkdTrain import ikd_train
from barcgp.prediction.InputGP.InputGPTrain import inputgp_train
import os

a_policy_name = 'aggressive_blocking'
a_policy_dir = os.path.join(train_dir, a_policy_name)
a_scencurve_dir = os.path.join(a_policy_dir, 'curve')
a_scenstraight_dir = os.path.join(a_policy_dir, 'straight')
a_scenchicane_dir = os.path.join(a_policy_dir, 'chicane')

t_policy_name = 'timid'
t_policy_dir = os.path.join(train_dir, t_policy_name)
t_scencurve_dir = os.path.join(t_policy_dir, 'curve')
t_scenstraight_dir = os.path.join(t_policy_dir, 'straight')
t_scenchicane_dir = os.path.join(t_policy_dir, 'chicane')

policy_name = 'IDK'
policy_dir = os.path.join(train_dir, policy_name)
sample_dir = os.path.join(policy_dir, 'sample')
ikd_scencurve_dir = os.path.join(policy_dir, 'curve')
ikd_scenstraight_dir = os.path.join(policy_dir, 'straight')
ikd_scenchicane_dir = os.path.join(policy_dir, 'chicane')

policy_name = 'race'
policy_dir = os.path.join(train_dir, policy_name)
track_scencurve_dir = os.path.join(policy_dir, 'track')


dirs = [track_scencurve_dir,a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir]
# ,t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]

ikd_dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir,
        t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir,
        ikd_scencurve_dir, ikd_scenstraight_dir, ikd_scenchicane_dir] 

def main():
    # print("1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("DynGP train init")
    # dyngp_train(dirs)
    # print("DynGP train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # print("2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("AutoEncoder train init")
    # encoder_train(dirs)
    # print("AutoEncoder train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # print("3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("IKD train init")
    # ikd_train(ikd_dirs)
    # print("IKD train Done")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("InputGP train init")
    inputgp_train(dirs)
    print("InputGP train Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    


if __name__ == "__main__":
    main()

