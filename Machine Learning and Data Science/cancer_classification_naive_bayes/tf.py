# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as s
import tensorflow as tf
from tensorflow import keras


#DATA Processing and Slicing
data = pd.read_csv("data.csv")
data.drop(labels=[data.columns[0],data.columns[32]],axis=1,inplace=True)
#data = data.iloc[:,[0,21,22]]
data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
training_data = data.iloc[0:int(0.7*len(data))]
remaining_data = data.iloc[int(0.7*len(data)):]
cv_data = remaining_data.iloc[0:int(0.33*len(remaining_data))]
testing_data = remaining_data.iloc[int(0.33*len(remaining_data)):]

x_train = training_data.iloc[:,[0]]
y_train = training_data.iloc[:,1:]
x_test = testing_data.iloc[:,[0]]
y_test = testing_data.iloc[:,1:]


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

model = keras.Sequential()

model.add(keras.layers.Dense(20,input_shape=(30,)))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(20,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(y_test,x_test, epochs=500,validation_split=0.3)

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']
plt.figure()

plt.plot(loss_values, 'bo', label='training loss')

plt.plot(val_loss_values, 'r', label = 'validation loss')