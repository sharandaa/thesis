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
from tensorflow import keras
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_classes = 30

test_data = pd.read_csv("/scratch/s2630575/thesis/labels/test_labels.csv")
true_labels = test_data['label'].tolist()

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    preprocessing_function=preprocess_input
)

test_set = test_datagen.flow_from_dataframe(
    dataframe=test_data,  # your training dataframe
    directory='/scratch/s2630575/thesis/test_AID',  # directory where your images are located
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

class_indices = test_set.class_indices
"""
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

model.load_weights("/scratch/s2630575/thesis/best_resnet.h5")
"""
model = keras.models.load_model("best_resnetmodel.h5")

prediction = model.predict(test_set)

prediction = np.argmax(prediction, axis = 1)

# Reverse the key-value pairs of the dictionary
labels_reverse = {v: k for k, v in class_indices.items()}

# Use the array elements as keys to retrieve their corresponding values (labels)
pred_labels = [labels_reverse[i] for i in prediction]

print(pred_labels)
print(true_labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# compute accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print('Accuracy:', accuracy)

# compute precision
precision = precision_score(true_labels, pred_labels, average='macro')
print('Precision:', precision)

# compute recall
recall = recall_score(true_labels, pred_labels, average='macro')
print('Recall:', recall)

# compute F1 score
f1 = f1_score(true_labels, pred_labels, average='macro')
print('F1 score:', f1)