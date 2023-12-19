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

# test_0 = get_dir('test_data', train_dir)

# timid_0 = get_dir('timid_0', train_dir)
# mild_5000_0 = get_dir('mild_5000_0', train_dir)
# reverse_0 = get_dir('reverse_0', train_dir)
# aggressive_blocking_0 = get_dir('aggressive_blocking_0', train_dir)



# wall_timid_0 = get_dir('wall_timid_0', train_dir)
# wall_mild_5000_0 = get_dir('wall_mild_5000_0', train_dir)
# wall_reverse_0 = get_dir('wall_reverse_0', train_dir)
# wall_aggressive_blocking_0 = get_dir('wall_aggressive_blocking_0', train_dir)

# dirs = timid.copy()
# # dirs.extend(m100)
# dirs.extend(m200)
# # dirs.extend(m300)
# dirs.extend(m500)
# # dirs.extend(m1000)
# dirs.extend(m5000)
# dirs.extend(reverse)

# dirs.extend(wall_timid)
# dirs.extend(wall_200)
# dirs.extend(wall_500)
# dirs.extend(wall_5000)
# dirs.extend(wall_reverse)

# dirs = test_0.copy()
# dirs = timid_0.copy()
# dirs.extend(mild_5000_0)
# dirs.extend(reverse_0)
# dirs.extend(aggressive_blocking_0)

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

timid = os.path.join(real_dir, 'lowspeed_nonblocking')
block = os.path.join(real_dir, 'lowspeed_blocking')
dirs = [timid,block]
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

