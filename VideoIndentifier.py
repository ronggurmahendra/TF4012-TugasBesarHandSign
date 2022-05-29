# import the opencv library
print("Importing Library...")
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2 
import os
import numpy as np
import scipy
from skimage import color, data, restoration
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import uniform
from tensorflow.keras.layers import BatchNormalization
# from object_detection.utils import label_map_util
import sys

# constants
modelPath = 'saved_model/Model_3'
width = 128
height = 128
dim = (width, height)
BATCH_SIZE = 32 
IMG_SIZE = (128, 128)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

print("Loading model from " + modelPath)
# define a video capture object
vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model(modelPath, compile = True)
print("Model Loaded")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

while True:
    ret, image_np = vid.read()

    reshaped_image = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)
    input_tensor = tf.convert_to_tensor(np.expand_dims(reshaped_image, 0), dtype=tf.float32)
    y_prob = model.predict(input_tensor, verbose=0)
    y_classes = y_prob.argmax(axis=-1)
    predicted_label = np.array(sorted(labels))[y_classes]
    print("Class : ", predicted_label, " | Class Index : ", y_classes,  " | Class Probability : ",y_prob.max())

    cv2.imshow('object detection', reshaped_image)

    # cv2.imwrite("./temp/temp.jpg", image_np)
    # cv2.imshow('object detection', image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
