#!/usr/bin/env python3
from predictor.common.utils.file_utils import *
import numpy as np
import gpytorch
from predictor.common.utils.scenario_utils import SampleGenerator, Sample
from predictor.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate
from predictor.prediction.gp_controllers import GPControllerApproximate


# policy_name = 'timid'
# policy_dir = os.path.join(train_dir, policy_name)
# timid_scencurve_dir = os.path.join(policy_dir, 'curve')
# timid_scenstraight_dir = os.path.join(policy_dir, 'straight')
# timid_scenchicane_dir = os.path.join(policy_dir, 'chicane')



# policy_name = 'wall'
# policy_dir = os.path.join(train_dir, policy_name)
# wall_scencurve_dir = os.path.join(policy_dir, 'sample')
# wall_scenstraight_dir = os.path.join(policy_dir, 'straight')
# wall_scenchicane_dir = os.path.join(policy_dir, 'chicane')

# policy_name = 'aggressive_blocking'
# policy_dir = os.path.join(train_dir, policy_name)
# scencurve_dir = os.path.join(policy_dir, 'curve')
# scenstraight_dir = os.path.join(policy_dir, 'straight')
# scenchicane_dir = os.path.join(policy_dir, 'chicane')


# policy_name = 'race'
# policy_dir = os.path.join(train_dir, policy_name)
# track_scencurve_dir = os.path.join(policy_dir, 'track')


# dirs = [wall_scencurve_dir, scencurve_dir] # , wall_scenstraight_dir,wall_scenchicane_dir , scencurve_dir,scenstraight_dir,  scenchicane_dir, timid_scencurve_dir,timid_scenstraight_dir, timid_scenchicane_dir]
# Training
def gp_main(dirs,realdata = False):
    
    # dirs = [track_scencurve_dir]
    # change curve type in file_utils
    sampGen = SampleGenerator(dirs, randomize=True,realdata = realdata)
    # sampGen.plotStatistics('s')
    # sampGen.even_augment('s', 0.3)
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=5)  # should be same as output_size
    gp_controller = GPControllerApproximate(sampGen, IndependentMultitaskGPModelApproximate, likelihood,
                                            input_size=11, output_size=5, inducing_points=200, enable_GPU=True)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood
    # gp_controller = GPControllerExact(sampGen, ExactGPModel, likelihood, 10, 5, True)

    gp_controller.train()
    gp_controller.set_evaluation_mode()
    trained_model = gp_controller.model, gp_controller.likelihood

    create_dir(path=model_dir)
    gp_name = 'gpberkely'
    gp_controller.save_model(gp_name)
    # gp_controller.evaluate()


# if __name__ == "__main__":
#     gp_main()
