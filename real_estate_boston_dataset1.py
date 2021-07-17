# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 17:36:29 2021

@author: akino

The objective is to use linear regression to find the median value of owner-occupied homes in 1000 USD's.
"""

import pandas as pd

from matplotlib import pyplot as plt

#reading the csv data
boston_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Boston_Housing/Training_set_boston.csv")
#split the dataset into input and output

X = boston_data.drop('MEDV', axis = 1)
y = boston_data.MEDV

# import train_test_split
from sklearn.model_selection import train_test_split

# Assign variables to capture train test split output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_features = X.shape[1]

#training the neural network model
from tensorflow.keras import Sequential # import Sequential from tensorflow.keras
from tensorflow.keras.layers import Dense # import Dense from tensorflow.keras.layers
from numpy.random import seed # seed helps you to fix the randomness in the neural network.
import tensorflow

# define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# import RMSprop optimizer
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop(0.1) # 0.01 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer) # compile the model

seed_value = 42
seed(seed_value) # If you build the model with given parameters, set_random_seed will help you produce the same result on multiple execution


# Recommended by Keras -------------------------------------------------------------------------------------
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# Recommended by Keras -------------------------------------------------------------------------------------


# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tensorflow.random.set_seed(seed_value)
model.fit(X_train, y_train, epochs=10, batch_size=30, verbose = 1) # fit the model

model.evaluate(X_test, y_test)