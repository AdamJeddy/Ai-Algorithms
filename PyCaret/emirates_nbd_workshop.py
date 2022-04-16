# -*- coding: utf-8 -*-
"""
Emirates NBD Workshop
"""

pip install pycaret

from pycaret.utils import enable_colab

enable_colab()

"""### Import data"""

from pycaret.datasets import get_data

dataset = get_data('credit')

# shape of the object
dataset.shape

data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

"""### Setting up Environment in PyCaret"""

from pycaret.classification import *

exp_clf101 = setup(data = data, target = 'default', session_id=123)

"""### Comparing the models """

best_model = compare_models()

print(best_model)

"""### Create a model"""

models()

"""### Decision Tree Classifier"""

dt = create_model('dt')

print(dt)

"""### K Neighbors Classifier"""

knn = create_model('knn')

"""###  Random Forest Classifier"""

rf = create_model('rf')

"""### Tune a Model

#### Decision Tree Classifier
"""

tuned_dt = tune_model(dt)

print(tuned_dt)

"""#### K Neighbors Classifier"""

import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})

print(tuned_knn)

"""#### Random Forest Classifier"""

tuned_rf = tune_model(rf)

