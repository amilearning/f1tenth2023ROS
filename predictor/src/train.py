import pickle
import os

directory_path = '/home/racepc/barc_data/realData'

file_list = os.listdir(directory_path)
pickle_file = None
for file in file_list:
    if file.endswith('.pkl'):
        pickle_file = file
        break

if pickle_file is not None:
    file_path = os.path.join(directory_path, pickle_file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # Do something with the loaded data
else:
    print("No pickle file found in the specified directory.")