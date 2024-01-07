import os
import random
import shutil

# Define source and destination directories
source_dir = "/home/hjpc/barc_data/trainingData/aggressive_train"
dest_dir = "/home/hjpc/barc_data/trainingData/aggressive_eval"

# Subfolders to process
subfolders = ["chicane", "curve", "straight"]
n_files = [40,40,20]
n_files = [int(itm/10) for itm in n_files]
           

for idx, subfolder in enumerate(subfolders):
    source_subfolder_path = os.path.join(source_dir, subfolder)
    dest_subfolder_path =  dest_dir # os.path.join(dest_dir, subfolder)

    # Ensure the destination subfolder exists
    os.makedirs(dest_subfolder_path, exist_ok=True)

    # List all files in the source subfolder
    files = os.listdir(source_subfolder_path)
    
    # Determine 20% of the files
    num_files_to_move = n_files[idx] # int(len(files) * 0.2)

    # Randomly select files
    selected_files = random.sample(files, num_files_to_move)

    # Move the selected files
    for file in selected_files:
        src_file_path = os.path.join(source_subfolder_path, file)
        dest_file_path = os.path.join(dest_subfolder_path, file)
        shutil.move(src_file_path, dest_file_path)

    print(f"Moved {num_files_to_move} files from {source_subfolder_path} to {dest_subfolder_path}")

print("File moving process completed.")
