#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction::

# Download dataset from this link:
# 
# https://www.kaggle.com/hellbuoy/car-price-prediction

# # Problem Statement::

# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# Which variables are significant in predicting the price of a car
# How well those variables describe the price of a car
# Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.
# 
# # task::
# We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# # WORKFLOW ::

# 1.Load Data
# 
# 2.Check Missing Values ( If Exist ; Fill each record with mean of its feature )
# 
# 3.Split into 50% Training(Samples,Labels) , 30% Test(Samples,Labels) and 20% Validation Data(Samples,Labels).
# 
# 4.Model : input Layer (No. of features ), 3 hidden layers including 10,8,6 unit & Output Layer with activation function relu/tanh (check by experiment).
# 
# 5.Compilation Step (Note : Its a Regression problem , select loss , metrics according to it)
# 6.Train the Model with Epochs (100) and validate it
# 
# 7.If the model gets overfit tune your model by changing the units , No. of layers , activation function , epochs , add dropout layer or add Regularizer according to the need .
# 
# 8.Evaluation Step
# 
# 9.Prediction

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset import section
dataset = pd.read_csv('CarPrice_Assignment.csv')

plt.figure(figsize=(25,20))
sns.countplot(dataset['carbody'])

#Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['CarName']=encoder.fit_transform(dataset['CarName'])
dataset['fuelsystem']=encoder.fit_transform(dataset['fuelsystem'])
dataset['cylindernumber']=encoder.fit_transform(dataset['cylindernumber'])
dataset['enginetype']=encoder.fit_transform(dataset['enginetype'])
dataset['enginelocation']=encoder.fit_transform(dataset['enginelocation'])
dataset['drivewheel']=encoder.fit_transform(dataset['drivewheel'])
dataset['carbody']=encoder.fit_transform(dataset['carbody'])
dataset['doornumber']=encoder.fit_transform(dataset['doornumber'])
dataset['aspiration']=encoder.fit_transform(dataset['aspiration'])
dataset['fueltype']=encoder.fit_transform(dataset['fueltype'])

plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(), annot = True, cmap = "RdYlGn")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 25].values

#Splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 7)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print("Train Accuracy:", regressor.score(x_train, y_train))
print("Test Accuracy:", regressor.score(x_test, y_test))


# In[ ]:




