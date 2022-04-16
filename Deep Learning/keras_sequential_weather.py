# -*- coding: utf-8 -*-
"""
Keras - Sequential - Weather
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=(224,224,3))
])

"""Change link according to the location on the drive"""

data_dir = '/map'

import pathlib
data_dir = pathlib.Path(data_dir)
data_dir

image_count = len(list(data_dir.glob('*/*.jpg')))+ len(list(data_dir.glob('*/*.jpeg')))+ len(list(data_dir.glob('*/*.png')))
image_count

map_img_dict = {
    'water' : list(data_dir.glob('water/*')),
    'desert': list(data_dir.glob('desert/*')),
    'cloudy': list(data_dir.glob('cloudy/*')),
    'green_area': list(data_dir.glob('green_area/*'))
}

map_label_dict = {
    'water' : 0,
    'desert': 1,
    'cloudy': 2,
    'green_area': 3
}

x,y = [], []

for map_image, images in map_img_dict.items():
    for image in images: 
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (224,224))
        x.append(resized_img)
        y.append(map_label_dict[map_image])

x = np.array(x);
y = np.array(y);

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = x_train/255
x_test= x_test/255

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(4))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'],label='vidaon accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test,  y_test)

tf.keras.utils.plot_model(model, show_shapes=True)

"""## Not the best model to use so try another"""

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

model = tf.keras.Sequential([ 
                    pretrained_model_without_top_layer, 
                    tf.keras.layers.Dense(4)
])
model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

model.fit(x_train, y_train, epochs=5)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'],label='vidaon accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test,  y_test)