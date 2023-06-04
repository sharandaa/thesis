import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from vit_keras import vit
import tensorflow_addons as tfa


print(os.getcwd())
# reading CSV file
data = pd.read_csv("/scratch/s2630575/thesis/labels/train_labels.csv")

# Split train set into train and validation sets
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=35)

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
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,  # your training dataframe
    directory='../train_all',  # directory where your images are located
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

class_indices = train_generator.class_indices
print(class_indices)
label_names = list(class_indices.keys())
print(label_names)

image_size = 256

vit_model = vit.vit_b16(
    image_size = image_size,
    activation = 'softmax',
    pretrained = True,
    include_top = False,
    pretrained_top = False,
    classes = num_classes)

# Add a global average pooling layer and a dense output layer
x = vit_model.output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(32, activation=tfa.activations.gelu)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=vit_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks to save the best model weights and stop training early if validation loss stops improving
checkpoint = ModelCheckpoint('best_vit.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

history = model.fit_generator(
        train_generator,
        steps_per_epoch= train_generator.samples // train_generator.batch_size, #len(train_generator),
        epochs=5,
        callbacks=[checkpoint, earlystop],
        validation_steps = validation_generator.samples // validation_generator.batch_size, #len(validation_generator),
        validation_data=validation_generator)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig("vitloss.pdf", format="pdf", bbox_inches="tight")

plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label ='val accuracy')
plt.legend()
plt.savefig("vitaccuracy.pdf", format="pdf", bbox_inches="tight")