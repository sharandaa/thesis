#import tensorflow as tf
import tensorflow_datasets as tfds
ds = tfds.load("tftest_AID", data_dir='/scratch/s2630575/tfds/tftest_AID')
print(ds)