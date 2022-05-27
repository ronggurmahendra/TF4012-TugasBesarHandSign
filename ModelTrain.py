# Import Library
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

# #Load Image

# images = []
# folder = "./Data/Images"
# for filename in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder,filename))
#     if img is not None:
#         images.append(img)
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
ClassifiedDataDir = "./Data/ClassifiedData/Train"
BATCH_SIZE = 8 
IMG_SIZE = (128, 128)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = ClassifiedDataDir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=1337,
        subset="training",
        validation_split=0.1,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        # rescale = 1./255,
    )
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = ClassifiedDataDir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=1337,
        subset="validation",
        validation_split=0.1,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        # rescale = 1./255,
    )
# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0
# dataset = ImageDataGenerator(
#     rescale = 1./255,
# )

# for element in dataset.as_numpy_iterator():
#     print(element)
class_names = train_dataset.class_names
print(class_names)
print(len(class_names))

# DataSet
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Model 2

input_size = 128
filter_size = 14
num_filter = 8
maxpool_size = 14
batch_size = BATCH_SIZE
epochs = 30

steps_per_epoch = 24720/batch_size

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(tf.keras.layers.Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=1))
model.add(tf.keras.layers.Dropout(uniform(0, 1)))

model.add(tf.keras.layers.Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(tf.keras.layers.Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))
model.add(tf.keras.layers.Dropout(uniform(0, 1)))  

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(24,activation='softmax'))

METRICS = [ 'accuracy']#, 'precision','recall']


model.compile( optimizer= tf.keras.optimizers.Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=METRICS)

#
model.summary()

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU')) 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Training
history = model.fit(    
    train_dataset,
    epochs=30,
    validation_data = (validation_dataset)
)

model.save('saved_model/model3UsingHandsignTopology')

loss, acc = model.evaluate(validation_dataset)
print(loss, acc)

print("Restored model, accuracy: {:5.2f}%".format(100 * acc))