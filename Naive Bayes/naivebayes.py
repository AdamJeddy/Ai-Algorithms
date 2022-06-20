# -*- coding: utf-8 -*-
"""
# Naive Bayes Classifiers
In machine learning, Naïve Bayes classification is a straightforward and powerful algorithm for the classification task. Naïve Bayes classification is based on applying Bayes’ theorem with strong independence assumption between the features. Naïve Bayes classification produces good results when we use it for textual data analysis such as Natural Language Processing.

Today's dataset contains information about adults with the target value of income.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('adult.csv', header=None)

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

df.head()

"""
Exercise 1

Display the information about the dataset. 
"""

df.info()

"""
Exercise 2 

How many categorical values do you have in the dataset? 
"""

intCol = df.count(numeric_only=True).count()
# Since there are only Int64 and Object Dtypes in the Dataset, I minused the total columns with count of int columns
len(df.columns) - intCol

"""
Exercise 3

How many sample of each label do you have in each column? (use for loop)
"""

# categ_col_names = ['workclass', 'education', 'marital_status', 
#                    'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
for i in df.columns:
    print(df[i].value_counts())
    print('\n')

"""
Exercise 4

Define your x, y and do a 75-25 training-testing split
"""

x = df.drop(columns=['income']) # All columns besides the first one
y = df.drop(df.loc[:, 'age':'native_country'].columns, axis = 1) # Only the first column

# Encode before splitting
import category_encoders as ce
ce_one_hot = ce.OneHotEncoder()
x = ce.OneHotEncoder().fit_transform(x, y)
y = ce.OrdinalEncoder().fit_transform(y, y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)
print('x test:  ', X_test.count()[1], ' | y test:  ', y_test.count()[0])
print('x train: ', X_train.count()[1], '| y train: ', y_train.count()[0])

"""
Exercise 5 

Use category encoder to encode each categorical column. 
"""

# I do this during the spliting phase, up in Exercise 4

"""#
Exercise 6 

Create and train your model.
"""

# instantiate the model
gnb = GaussianNB()

gnb.fit(X_train, np.ravel(y_train))

"""
Exercise 7

Display the accuracy (testing) and verify there is no overfitting. 
Additionally compare your results with a null classifer. 
A null classifier is when you always predict the most frequent target. 
"""

y_pred = gnb.predict(X_test)
print("Test Accuracy:", gnb.score(y_test, y_pred))

y_pred = gnb.predict(X_train)
print("Training Accuracy:", gnb.score(y_train, y_pred))

# The accuracy isnt that high so there isnt any overfitting.

# Null classifier, again the percenage isnt that high when compared to the other accuracy so there is no overfitting
y_test.value_counts().head(1) / len(y_test)

"""
Improve the accuracy, use anything you know to improve the performance 
"""

df2 = df.copy()

# Preprocessing
df2['education'] = df2['education'].replace([' 11th', ' 10th', ' 9th', ' 7th-8th', ' 12th', 
                                             ' 5th-6th', ' 1st-4th', ' Preschool', ' Prof-school'], ' Didn\'t pass School')
df2['education'] = df2['education'].replace([' Assoc-voc', ' Assoc-acdm'], ' Assoc')
df2['marital_status'] = df2['marital_status'].replace([' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse'], ' Married')
#df2['education'].value_counts()

df2['income'].value_counts()

# Fixing undersampling
df2_1 = df2.loc[df2['income'] == ' <=50K'][:7800]
df2_2 = df2.loc[df2['income'] == ' >50K'][:7800]

df2 = df2_1.append(df2_2).reset_index(drop=True)
df2['income'].value_counts()

x = df2.drop(columns=['income']) # All columns besides the first one
y = df2.drop(df2.loc[:, 'age':'native_country'].columns, axis = 1) # Only the first column
ce_one_hot = ce.OneHotEncoder()
x = ce.OneHotEncoder().fit_transform(x, y)
y = ce.OrdinalEncoder().fit_transform(y, y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)
print('x test:  ', X_test.count()[1], ' | y test:  ', y_test.count()[0])
print('x train: ', X_train.count()[1], '| y train: ', y_train.count()[0])

gnb = GaussianNB()
gnb.fit(X_train, np.ravel(y_train))

y_pred = gnb.predict(X_test)
print("Test Accuracy:", gnb.score(y_test, y_pred))

y_pred = gnb.predict(X_train)
print("Training Accuracy:", gnb.score(y_train, y_pred))

y_test.value_counts().head(1) / len(y_test)

"""
Another method
**Decision Tree Doesn't help improve the accuracy**
"""

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

y_pred = dtree.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred))
