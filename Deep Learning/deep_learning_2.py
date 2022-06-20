# -*- coding: utf-8 -*-
"""
# Deep Learning
Deep learning is a specialized form of machine learning. A machine learning workflow starts with relevant features being manually extracted from images. The features are then used to create a model that categorizes the objects in the image. With a deep learning workflow, relevant features are automatically extracted from images. In addition, deep learning performs “end-to-end learning” – where a network is given raw data and a task to perform, such as classification, and it learns how to do this automatically.

Another key difference is deep learning algorithms scale with data, whereas shallow learning converges. Shallow learning refers to machine learning methods that plateau at a certain level of performance when you add more examples and training data to the network.

A key advantage of deep learning networks is that they often continue to improve as the size of your data increases. In machine learning, you manually choose features and a classifier to sort images. With deep learning, feature extraction and modeling steps are automatic.

## Exercise 1: Design simple classifiers for iris dataset
Use the following libraries below. No need for normalization.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import load_iris

import tensorflow as tf

df = pd.read_csv('data.data')
X = df.drop('g', axis=1)
y = df['g']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#iris = datasets.load_iris()
#cols_name = iris.feature_names
#X = iris.data
#y = iris.target

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Holds all the model scores 
clf_scores = []

"""### Classifier 1: Random Forest"""

# Initializing the model 
clf_randForest = RandomForestClassifier()

# Fitting data to the model
clf_randForest.fit(X_train,y_train)

# Predicting and Model Accuracy
y_pred = clf_randForest.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

clf_scores.append(['Random Forest Classifier', accuracy_score(y_test, y_pred)])

"""###  Classifier 2: Logistic Regression"""

# Initializing the model
clf_LogisticReg = LogisticRegression()

# Fitting data to the model
clf_LogisticReg.fit(X_train,y_train)

# Model Accuracy
print("Accuracy: ", clf_LogisticReg.score(X_test, y_test))

clf_scores.append(['Logistic Regression', clf_LogisticReg.score(X_test, y_test)])

"""### Classifier 3: Decision Tree Classifier"""

# Initializing the model
clf_dtc = DecisionTreeClassifier()

# Fitting data to the model
clf_dtc.fit(X_train,y_train)

# Predicting and Model Accuracy
y_pred = clf_dtc.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

clf_scores.append(['Decision Tree Classifier', accuracy_score(y_test, y_pred)])

"""### Classifier 4: K-Neighbors Classifier"""

# Initializing the model
clf_kn = KNeighborsClassifier()

# Fitting data to the model
clf_kn.fit(X_train, y_train)

# Predicting and Model Accuracy
y_pred = clf_kn.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

clf_scores.append(['K-Neighbors Classifier', accuracy_score(y_test, y_pred)])

"""### Classifier 5: Support Vector Classification (SVC)"""

# Initializing the model
clf_svc = SVC()

# Fitting data to the model
clf_svc.fit(X_train, y_train)

# Predicting and Model Accuracy
y_pred = clf_svc.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

clf_scores.append(['SVC', accuracy_score(y_test, y_pred)])

"""### Classifier 6: Gaussian NB"""

# Initializing the model
clf_gnb = GaussianNB()

# Fitting data to the model
clf_gnb.fit(X_train, y_train)

# Predicting and Model Accuracy
y_pred = clf_gnb.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

clf_scores.append(['GaussianNB', accuracy_score(y_test, y_pred)])

"""### All the classifiers and their score"""

clf_scores

"""## Exercise 2: Neural Network
Now design a neural network to classify the iris flowers.


### step 1: Import the following libraries
"""

from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

"""### Step 2: Set up your data as x and y. Split to train and test (70-30).
You may want to encode your targets before splitting.
"""

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""### Step 3: Build the network structure 
Model will be sequential, with input shape same as the number of features. Use relu activation function. 
The last layer must be a softmax layer with N nodes. where N is the number of classes. 
"""

# Initilizing the model
model = Sequential()

# Adding layers to the model
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

"""### Step 4: Model parameters 
Set your optimizer to Adam with a learn rate of 0.001 or 1e-3.
compile your model, set your loss as categorical_crossentropy and your metrics as accuracy. 
you can display your network by checking the summary() function. 
"""

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

"""### Step 5: Fit and display the accuracy"""

# Training the model
model.fit(X_train, y_train, verbose=2, batch_size=5, epochs=200)

# Test the model with unseen data
results = model.evaluate(X_test, y_test)
print('Final test set loss:\t{:4f}'.format(results[0]))
print('Final train set loss:\t{:4f}'.format(results[1]))

