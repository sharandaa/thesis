import os
import numpy as np
import tensorflow as tf
import pandas as pd
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, AveragePooling2D, BatchNormalization
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow import keras
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from vit_keras import vit
import tensorflow_addons as tfa

testimg = {"/scratch/s2630575/thesis/test_AID":"/scratch/s2630575/thesis/labels/test_labels.csv",
           "/home/s2630575/espcnres/ESPCN_real_x2":"/scratch/s2630575/thesis/labels/test_labels.csv", 
           "/home/s2630575/SwinIR/results/swinir_real_sr_x2":"/scratch/s2630575/labels/test_labels_swinir.csv", 
           "/home/s2630575/swin2sr/results/swin2sr_real_sr_x4":"/scratch/s2630575/labels/test_labels_swin2sr.csv", 
           "/home/s2630575/Real-ESRGAN/results/realx2":"/scratch/s2630575/labels/test_labels_realesrgan.csv",
           "/scratch/s2630575/datasets/finetuned_espcn":"/scratch/s2630575/labels/test_labels_finetuned_espcn.csv"}

testimg = {"/scratch/s2630575/datasets/finetuned_espcn":"/scratch/s2630575/thesis/labels/test_labels.csv"}

num_classes = 30

for key, value in testimg.items():
    print(key)
    test_data = pd.read_csv(value)
    true_labels = test_data['label'].tolist()

    test_datagen = ImageDataGenerator(
        rescale = 1./255,
    )

    test_set = test_datagen.flow_from_dataframe(
        dataframe=test_data,  # your training dataframe
        directory=key,  # directory where your images are located
        x_col='filename',
        y_col='label',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle = False
    )

    class_indices = test_set.class_indices

    image_size = 256

    vit_model = vit.vit_b32(
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

    model.load_weights("/scratch/s2630575/thesis/best_vit36.h5")

    prediction = model.predict(test_set)

    predictiontop1 = np.argmax(prediction, axis = 1)

    # Reverse the key-value pairs of the dictionary
    labels_reverse = {v: k for k, v in class_indices.items()}

    # Use the array elements as keys to retrieve their corresponding values (labels)
    pred_labels = [labels_reverse[i] for i in predictiontop1]

    #print(pred_labels)
    #print(true_labels)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

    from sklearn.metrics import top_k_accuracy_score

    #prediction = model.predict(test_set)
    predictiontop5 = np.argsort(prediction, axis=1)[:, ::-1]  # Sort the predictions in descending order

    # Assuming actual_labels contains the true labels for the test set
    true_labelsindex = test_set.classes  # Replace with your actual labels

    # Calculate the top 5 accuracy
    top5_accuracy = top_k_accuracy_score(true_labelsindex, prediction, k=5)
    print('top 5 accuracy:', top5_accuracy)

    #class_report = classification_report(true_labels, pred_labels)
    #print(class_report)