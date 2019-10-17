#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:41:19 2019

@author: shitiz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data-Set
dataset=pd.read_csv('train.csv')
dataset_test=pd.read_csv('test.csv')
X=dataset.iloc[:,[2,4,5,6,7]].values
X_test=dataset_test.iloc[:,[1,3,4,5,6]].values
Y=dataset.iloc[:,1].values

# Pre-Processing

# For missing data
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values="NaN",strategy="mean")
imputer_2 =Imputer(missing_values="NaN",strategy="mean")
imputer=imputer.fit(X[:,2:3])
imputer_2=imputer_2.fit(X_test[:,2:3])
X[:,2:3]=imputer.transform(X[:,2:3])
X_test[:,2:3]=imputer_2.transform(X_test[:,2:3])



#Cateogorical_Features

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
labelencoder_X_2=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X_test[:,1]=labelencoder_X_2.fit_transform(X_test[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
onehotencoder_2=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X_test=onehotencoder_2.fit_transform(X_test).toarray()

X=X[:,1:6]
X_test=X_test[:,1:6]

# ANN

# Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(activation="relu", input_dim=5, units=3, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=3, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x=X,y=Y,batch_size=4,epochs=100)

y_pred=classifier.predict(X_test)

from pandas import DataFrame



for i in range(0,418):
   y_pred[i]=y_pred[i]>0.5
   
Z=pd.DataFrame(data=dataset_test.iloc[:,0].values,columns=['PassengerId'])

outputframe=pd.DataFrame(data=y_pred,columns=['Survived']) 
outputframe=pd.concat([Z,outputframe],axis=1)





outputframe.to_csv('output.csv')  




