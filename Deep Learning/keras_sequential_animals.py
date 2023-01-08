# -*- coding: utf-8 -*-
"""
Keras - Sequential - Animals
"""

import tensorflow as tf
import numpy as np 
import cv2 
import PIL.Image as Image
import matplotlib.pylab as plt 
import tensorflow_hub as hub

#from google.colab import drive
#drive.mount('/content/drive')

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=(224,224,3))
])

ImageSize= (224,224)
myImage = Image.open("komodo.jpg").resize(ImageSize)
myImage

myImage = np.array(myImage)/255

myImage[np.newaxis,...]

predicted_label = classifier.predict(myImage[np.newaxis,...])

predicted_label

predicted_label= np.argmax(predicted_label)
predicted_label

image_labels = []
with open("ImageNetLabels.txt", "r") as f:
    image_labels = f.read().splitlines()
image_labels[40:50]

image_labels[predicted_label]

data_dir = "/animals"

import pathlib
data_dir = pathlib.Path(data_dir)
data_dir

image_count = len(list(data_dir.glob('*/*.jpg')))+ len(list(data_dir.glob('*/*.jpeg')))+ len(list(data_dir.glob('*/*.png')))
image_count

animal_img_dict = {
    'Dog' : list(data_dir.glob('Dog/*')),
    'Cat': list(data_dir.glob('Cat/*')),
    'Tiger': list(data_dir.glob('Tiger/*')),
    'Rabbit': list(data_dir.glob('Rabbit/*'))
}

animal_label_dict = {
    'Dog': 0,
    'Cat': 1,
    'Tiger': 2,
    'Rabbit': 3
}

x,y = [], []

for animal_image, images in animal_img_dict.items():
    for image in images: 
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (224,224))
        x.append(resized_img)
        y.append(animal_label_dict[animal_image])

x = np.array(x)
y = np.array(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = x_train/255
x_test= x_test/255

model = tf.keras.models.Sequential()
from tensorflow.keras import layers

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

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_of_animals = 4

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_animals)
])

model.summary()

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'],label='vidaon accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

y_pred = np.argmax(probability_model.predict(x_test),axis=1)
y_pred

y_test
cm=confusion_matrix(y_test,y_pred)
cm

class_names=['Dog', 'Cat', 'Tiger', 'Rabbit']

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# In[34]:


import itertools
plot_confusion_matrix(cm,class_names)

fileIdx= 31
plt.imshow(x_test[fileIdx])

result = probability_model.predict(np.array([x_test[fileIdx]]))
result = np.argmax(result)
result

class_names[result]

y_test[fileIdx]



