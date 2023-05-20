import os
import pandas as pd

#img_dir = r"C:\Users\shara\OneDrive\Documents\Scriptie\datasets\train_AID"
img_dir = "/home/s2630575/swin2sr/results/swin2sr_real_sr_x4"

label_dict = {'filename': [], 'label': []}

for filename in os.listdir(img_dir):
    label = filename.split('_')[0]  # Extract the label from the filename
    label_dict['filename'].append(filename)
    label_dict['label'].append(label)

df = pd.DataFrame.from_dict(label_dict)

print(df.head())
#print(df)

df.to_csv("/scratch/s2630575/labels/test_labels_swin2sr.csv", index = False)