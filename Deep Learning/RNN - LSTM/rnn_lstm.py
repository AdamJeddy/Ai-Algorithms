# -*- coding: utf-8 -*-
"""
# Deep Learning - Recurrent Neural Networks

Implementation of a simple LSTM network that tries to estimate the google stock prices given a sequence of data.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""## 1. Import your datasets train and test """

train = pd.read_csv('trainset.csv')
train.head()

test = pd.read_csv('testset.csv')
test.head()

"""## 2. Separate your open column and store it in a variable. 
Training can only be done on numpy arrays, therefore we have to transform the dataframe into a numpy array.
"""

# Seperating the col and changing it to a numpy array and resizing it 
training = train['Open'].to_numpy().reshape(-1, 1) 
type(training)

"""## 3. Use a MaxMinScaler and scale your data to a range of 0-1."""

# import
from sklearn.preprocessing import MinMaxScaler

# initialize 
scaler = MinMaxScaler(feature_range=(0,1))

# Scale the data
training_scaled = scaler.fit_transform(training)
training_scaled

"""## 4. Create empty arrays for x, y of train and test set. 
We will use windows of 60 timestaps to predict the 61st sample. Use a for loop, that ranges to length of training or testing file. Every 60 sample, append to your training set. 
Keep in mind that labels: 

        x_train.append(training_scaled[i-60:i, 0])
        y_train.append(training_scaled[i,0])
"""

x_train = []
y_train = []

for i in range(60, len(training_scaled)):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

"""## 5. reshape your data such that it has space for another set of features. """

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

"""**Training and testing files should be ready**  """

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

"""## 6. Create a regressor model that has the following structure. 
![image.png](attachment:image.png)
"""

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

"""## 7. Compile your model using the adam optimizer and set your losses for 'mean_squared_error'. and fit your data with 75 epochs."""

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(model.summary())

model.fit(x_train, y_train, epochs = 75)

"""## 8. Concatenate your train['open'] and test['open'] with axis =0. 

"""

testing = test['Open'].to_numpy().reshape(-1, 1)

concat_open = pd.concat((test['Open'], train['Open']), axis=0)

"""## 9. Make sure your inputs start from index 60. reshape them into a single column and apply the scaler transform. """

inputs = concat_open[1259 - 60:].to_numpy().reshape(-1, 1)

inputs = scaler.fit_transform(inputs)
inputs

"""## 10. Refer to step 4, if you have completed it for x_test move to step 11, else append your data in x_test in the same way. """

x_test = []

for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])

"""## 11. Convert to a numpy array and reshape similar to step 5."""

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_test.shape

"""## 12. Predict your results and plot them against the real values."""

pred = model.predict(x_test)

# undo the transformation by inverse-transforming since we need it in the original form
pred = scaler.inverse_transform(pred)

plt.plot(testing, label='Stock Price')
plt.plot(pred, label='Prediction')

plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()