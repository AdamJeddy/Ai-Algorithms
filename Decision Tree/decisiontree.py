# -*- coding: utf-8 -*-
"""
## Decision Tree

### Import your dataset
This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms. The dataset has one binary class (i.e., the first column) and 22 attributes (i.e., all other columns) and contains 8124 records. In this task, you can treat the missing values as a new category of the attribute.
https://archive.ics.uci.edu/ml/datasets/Mushroom

### built-in functions and attributes 
Make use of the function imported below.
"""

# import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix

data= pd.read_csv('mushrooms.csv')
data.head()

data.info()

data.describe()

""" 1. Count the number of samples in each class"""

data.groupby("class").size()
#data.value_counts('class') # Alternative solution

""" 2. Check if you have any null values """

data.isnull().sum()

""" 3.Use bar chart to plot class e and p (use seaborn)"""

sns.countplot(x="class", data=data)

""" 4. Convert the categorical values to numberical using LabelEncoder"""

df = data.apply(LabelEncoder().fit_transform)
df.head()

""" 5.Define your x and y. Split them into 70-30 % for training."""

x = df.drop(columns=['class']) # All columns besides the first one
y = df.drop(data.loc[:, 'cap-shape':'habitat'].columns, axis = 1) # Only the first column

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

""" 6. Create your decision tree model and train """

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

""" 7. Display Training Accuracy, Testing Accuracy and produce """

y_pred = dtree.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

y_pred = dtree.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred))

""" 8. Plot the confusion matrix and get the classification report"""

tn, fp, fn, tp = confusion_matrix(y_test, dtree.predict(X_test)).ravel()
print('True Negative\t= ', tn, '\nFalse Positive\t= ', fp, '\nFalse Negative\t= ', fn, '\nTrue Positive\t= ', tp)

plot_confusion_matrix(dtree, X_test, y_test)

print(classification_report(y_train, y_pred))
