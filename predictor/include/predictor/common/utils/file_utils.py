import os
import pickle
import pathlib

import sys
sys.path.append("/home/orin1/gp-opponent-prediction-models/barcgp")

gp_dir = os.path.expanduser('~') + '/barc_data/'
train_dir = os.path.join(gp_dir, 'trainingData/')
real_dir = os.path.join(gp_dir, 'realData/')
multiEval_dir = os.path.join(gp_dir, 'MultiEvalData/')
eval_dir = os.path.join(gp_dir, 'evaluationData/')
model_dir = os.path.join(train_dir, 'models/')
param_dir = os.path.join(gp_dir, 'params/')
track_dir = os.path.join(gp_dir, 'tracks/')
static_dir = os.path.join(gp_dir, 'statics/')
fig_dir = os.path.join(gp_dir, 'figures/')

def dir_exists(path=''):
    dest_path = pathlib.Path(path).expanduser()
    return dest_path.exists()


def create_dir(path='', verbose=False):
    dest_path = pathlib.Path(path).expanduser()
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
        return dest_path
    else:
        if verbose:
            print('- The source directory %s does not exist, did not create' % str(path))
        return None



def pickle_write(data, path):
    # Extract the directory from the path
    directory = os.path.dirname(path)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Write the data to the file
    with open(path, 'wb') as dbfile:
        pickle.dump(data, dbfile)


def pickle_read(path):
    print("path = "+ str(path))
    dbfile = open(path, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data
