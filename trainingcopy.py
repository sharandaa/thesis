import os
import shutil

# Set the directory paths
folder1 = '../../thesis_train/train_AID'
folder2 = '../../thesis_train2/train_AID2'
output_folder = '../../train_all'

# Loop through the files in the first folder
for filename in os.listdir(folder1):
    # Copy the file to the output folder
    shutil.copy(folder1 + "/" + filename, output_folder + "/" + filename)

# Loop through the files in the second folder
for filename in os.listdir(folder2):
    # Copy the file to the output folder
    shutil.copy(folder2 + "/" + filename, output_folder +"/" + filename)