import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, AveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed_num = 35

import random as rn
from tensorflow.compat.v1.keras import backend as K
# https://stackoverflow.com/questions/61368342/how-can-i-get-reproducible-results-in-keras-for-a-convolutional-neural-network-u
#tf.keras.backend.clear_session()

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed_num)
rn.seed(seed_num)
tf.random.set_seed(seed_num)
#session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
#K.set_session(sess)


# reading CSV file
data = pd.read_csv("/scratch/s2630575/thesis/labels/train_labels.csv")

# Split train set into train and validation sets
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=seed_num)

num_classes = len(data['label'].unique())

# set up data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

# load images from directory and create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,  # your training dataframe
    directory='../train_all',  # directory where your images are located
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    seed=seed_num
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,  # your training dataframe
    directory='../train_all',  # directory where your images are located
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    seed=seed_num
)

class_indices = train_generator.class_indices
print(class_indices)
label_names = list(class_indices.keys())
print(label_names)

# Set up the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense output layer
x = base_model.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks to save the best model weights and stop training early if validation loss stops improving
checkpoint = ModelCheckpoint('best_resnet35.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

history = model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples // train_generator.batch_size, #len(train_generator),
        epochs=50,
        callbacks=[checkpoint, earlystop],
        validation_steps = validation_generator.samples // validation_generator.batch_size, #len(validation_generator),
        validation_data=validation_generator)

"""
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig("resnet/resnetloss.pdf", format="pdf", bbox_inches="tight")

plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label ='val accuracy')
plt.legend()
plt.savefig("resnet/resnetaccuracy.pdf", format="pdf", bbox_inches="tight")
"""