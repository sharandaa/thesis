#import tensorflow as tf
import tensorflow_datasets as tfds
ds = tfds.load("aidtestx4", data_dir='/scratch/s2630575/tfds')
print(ds)