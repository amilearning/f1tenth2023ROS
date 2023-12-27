from predictor.common.utils.file_utils import *
from predictor.prediction.cont_encoder.cont_encoderTrain import cont_encoder_train
from predictor.prediction.thetaGP.ThetaGPTrain import thetagp_train
from predictor.prediction.gp_berkely_train import gp_main
from predictor.prediction.covGP.covGPNN_Train import covGPNN_train, tsne_analysis
import os

def get_dir(policy_name_, train_dir_):
    policy_dir = os.path.join(train_dir_, policy_name_)
    scencurve_dir = os.path.join(policy_dir, 'curve')
    scenstraight_dir = os.path.join(policy_dir, 'straight')
    scenchicane_dir = os.path.join(policy_dir, 'chicane')
    dirs = [scencurve_dir, scenstraight_dir, scenchicane_dir]
    return dirs


tsne_timid_0 = get_dir('tsne_timid_0', train_dir)
tsne_mild_5000_0 = get_dir('tsne_mild_5000_0', train_dir)
tsne_reverse_0 = get_dir('tsne_reverse_0', train_dir)
tsne_aggressive_blocking_0 = get_dir('tsne_aggressive_blocking_0', train_dir)


timid_0 = get_dir('timid_0', train_dir)
aggressive_blocking_0 = get_dir('aggressive_blocking_0', train_dir)
mild_5000_0 = get_dir('mild_5000_0', train_dir)
reverse_0 = get_dir('reverse_0', train_dir)



dirs = timid_0.copy()
dirs.extend(aggressive_blocking_0)
dirs.extend(mild_5000_0)
dirs.extend(reverse_0)

# dirs.extend(wall_timid_0)
# dirs.extend(wall_mild_5000_0)
# dirs.extend(wall_reverse_0)
# dirs.extend(wall_aggressive_blocking_0)   


# dirs = timid.copy()

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

# policy_name = 'race'
# policy_dir = os.path.join(train_dir, policy_name)
# track_scencurve_dir = os.path.join(policy_dir, 'track')


# policy_name = 'wall'
# policy_dir = os.path.join(train_dir, policy_name)
# wall_scencurve_dir = os.path.join(policy_dir, 'curve')
# wall_scenstraight_dir = os.path.join(policy_dir, 'straight')
# wall_scenchicane_dir = os.path.join(policy_dir, 'chicane')

# dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir]
# dirs = [a_scencurve_dir, a_scenstraight_dir, a_scenchicane_dir,t_scencurve_dir, t_scenstraight_dir, t_scenchicane_dir]
# dirs = [track_scencurve_dir]


def main():  


    tsne_analysis(dirs, load_data=False)

  


if __name__ == "__main__":
    main()

