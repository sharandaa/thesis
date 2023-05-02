# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:57:24 2023

@author: shara
"""

import os
import random
import shutil
from sklearn.model_selection import train_test_split

source = "C:/Users/shara/OneDrive/Documents/Scriptie/AID"

# Source folders containing pictures
src_folders = [source + "/AID/Airport", source + "/AID/BareLand", source + "/AID/BaseballField", source + "/AID/Beach",
               source + "/AID/Bridge", source + "/AID/Center", source + "/AID/Church", source + "/AID/Commercial",
               source + "/AID/DenseResidential", source + "/AID/Desert", source + "/AID/Farmland", source + "/AID/Forest",
               source + "/AID/Industrial", source + "/AID/Meadow", source + "/AID/MediumResidential", source + "/AID/Mountain",
               source + "/AID/Park", source + "/AID/Parking", source + "/AID/Playground", source + "/AID/Pond", source + "/AID/Port",
               source + "/AID/RailwayStation", source + "/AID/Resort", source + "/AID/River", source + "/AID/School",
               source + "/AID/SparseResidential", source + "/AID/Square", source + "/AID/Stadium", source + "/AID/StorageTanks",
               source + "/AID/Viaduct"]

# Destination folders for selected pictures
dest_folders = [source + "/mod_AID/Airport", source + "/mod_AID/BareLand", source + "/mod_AID/BaseballField", source + "/mod_AID/Beach",
               source + "/mod_AID/Bridge", source + "/mod_AID/Center", source + "/mod_AID/Church", source + "/mod_AID/Commercial",
               source + "/mod_AID/DenseResidential", source + "/mod_AID/Desert", source + "/mod_AID/Farmland", source + "/mod_AID/Forest",
               source + "/mod_AID/Industrial", source + "/mod_AID/Meadow", source + "/mod_AID/MediumResidential", source + "/mod_AID/Mountain",
               source + "/mod_AID/Park", source + "/mod_AID/Parking", source + "/mod_AID/Playground", source + "/mod_AID/Pond",
               source + "/mod_AID/RailwayStation", source + "/mod_AID/Resort", source + "/mod_AID/River", source + "/mod_AID/School",
               source + "/mod_AID/SparseResidential", source + "/mod_AID/Square", source + "/mod_AID/Stadium", source + "/mod_AID/StorageTanks",
               source + "/mod_AID/Viaduct"]

dest_folder = [source + "/datasets/dataset1", source + "/datasets/dataset2", source + "/datasets/dataset3", 
               source + "/datasets/dataset4", source + "/datasets/dataset5"]

dest = "C:/Users/shara/OneDrive/Documents/Scriptie/AID/AID/All"

# Number of pictures to select from each source folder
num_pics = 50

# Create destination folders if they don't exist
"""
for dest_folder in dest_folders:
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
"""
def combine():
    # Loop over each source folder and destination folder pair
    for i in range(len(src_folders)):
        # Get a list of all pictures in the source folder
        pics_list = [f for f in os.listdir(src_folders[i]) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
        
        for pic in pics_list:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest, pic)
            shutil.copy(src_path, dest_path)

        """
        # Select random pictures from the list
        selected_pics = random.sample(pics_list, num_pics)

        data1 = selected_pics[:10]
        data2 = selected_pics[10:20]
        data3 = selected_pics[20:30]
        data4 = selected_pics[30:40]
        data5 = selected_pics[40:50]
        
        # Copy selected pictures to destination folder
        for pic in data1:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest_folder[0], pic)
            shutil.copy(src_path, dest_path)

        for pic in data2:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest_folder[1], pic)
            shutil.copy(src_path, dest_path)
        
        for pic in data3:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest_folder[2], pic)
            shutil.copy(src_path, dest_path)
        
        for pic in data4:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest_folder[3], pic)
            shutil.copy(src_path, dest_path)
        
        for pic in data5:
            src_path = os.path.join(src_folders[i], pic)
            dest_path = os.path.join(dest_folder[4], pic)
            shutil.copy(src_path, dest_path)
        #print(f"{num_pics} pictures have been randomly selected and copied from {src_folder} to {dest_folder}.")
        """

print("Done!")

src_all = "C:/Users/shara/OneDrive/Documents/Scriptie/AID/AID/All"
train_folder = ""
test_folder = "C:/Users/shara/OneDrive/Documents/Scriptie/AID/AID/test_AID"

def split():
    pics_list = [f for f in os.listdir(src_all) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
    
    # split the list into training and testing sets
    train_filenames, test_filenames = train_test_split(pics_list, test_size=0.1, random_state=58, shuffle = True)

    # print the training and testing sets
    print('Training set:', train_filenames)
    print('Testing set:', test_filenames)

    for pic in test_filenames:
        src_path = os.path.join(src_all, pic)
        dest_path = os.path.join(test_folder, pic)
        shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    split()