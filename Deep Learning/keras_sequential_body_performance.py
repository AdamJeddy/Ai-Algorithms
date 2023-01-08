# -*- coding: utf-8 -*-
"""
Keras - Sequential - Body Performance

Imports
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
from tensorflow.keras import layers

"""### Import Data"""

df = pd.read_csv('bodyPerformance.csv')

df.head()

df.info()

df.describe()

class_names = ['1', '2', '3', '4']

enc = OrdinalEncoder()
temp = enc.fit_transform(df)
temp = pd.DataFrame(temp)

features = temp.copy()
labels = features[features.columns[:-1]]

features.head()

labels.head()

features_arr = np.array(features)
features_arr

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(11)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(), 
              optimizer = tf.optimizers.Adam(), 
              metrics=['accuracy'])

features_arr = np.asarray(features_arr).astype(np.float32)

model.fit(features_arr, labels, epochs=15)

normalize = layers.Normalization()

normalize.adapt(features_arr)

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(11)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                           optimizer = tf.optimizers.Adam(),
                           metrics=['accuracy'])

norm_abalone_model.fit(features_arr, labels, epochs=15)

"""Its a better accuracy now"""
