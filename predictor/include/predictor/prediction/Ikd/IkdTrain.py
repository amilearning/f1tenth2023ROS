#!/usr/bin/env python3

from barcgp.common.utils.file_utils import *
import numpy as np
import torch
from barcgp.common.utils.scenario_utils import SampleGeneartorIKDTime, SampleGenerator, Sample
from barcgp.prediction.Ikd.IkdPredictor import IkdPredictor

from torch.utils.data import DataLoader, random_split

# policy_name = 'IDK'
# policy_dir = os.path.join(train_dir, policy_name)
# sample_dir = os.path.join(policy_dir, 'sample')
# scencurve_dir = os.path.join(policy_dir, 'curve')
# scenstraight_dir = os.path.join(policy_dir, 'straight')
# scenchicane_dir = os.path.join(policy_dir, 'chicane')

# Training
def ikd_train(dirs):
    # dirs = [sample_dir] #[scencurve_dir, scenstraight_dir, scenchicane_dir]    
    
    sampGen = SampleGeneartorIKDTime(dirs, randomize=True)
    
    # sampGen.plotStatistics()
    
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")


    args_ =  {
            "batch_size": 128,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": sampGen.input_dim,
            "hidden_size": 20,
            "latent_size": 2,
            "output_size": sampGen.output_dim,
            "learning_rate": 0.001,
            "max_iter": 50000,
            "sequence_length":sampGen.horizon_length
            }
    idkpred = IkdPredictor(args= args_, model_type = "ConvFFNN")
    
    
    # train_x, train_y, test_x, test_y = sampGen.getTrainingAndTestData()
    train_dataset, val_dataset, test_dataset  = sampGen.get_datasets()

    batch_size = args_["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    idkpred.set_train_loader(train_loader)
    idkpred.set_test_loader(test_loader)

    # idkpred.set_train_data(train_x,train_y)
    # idkpred.set_test_data(test_x,test_y)
    idkpred.train(args= args_)
    create_dir(path=model_dir)
    idkpred.model_save()
    # idkpred.evaluate()


# if __name__ == "__main__":
#     main()
