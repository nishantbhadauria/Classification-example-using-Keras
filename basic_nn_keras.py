# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:22:30 2019

@author: Nishant
"""

import pandas as pd
heart=pd.read_csv('C:\Kaggle\heart.csv')


""" Renaming the variables to better understand """

heart.columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

"""Generic correlation to find what impacts the heart attack"""

heart.corr()

"""Visualization by classes to better understand the factors"""
#taking only numeric columns
heart_num=heart[['age','resting_blood_pressure','cholesterol','st_depression','target','max_heart_rate_achieved']]
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(heart_num, hue="target")

"""min max scaling(converting evrything between 0 and 1 ..basic data prep for nn"""
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler().fit(heart[['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']])
heart_minMax=minmax_scale.transform(heart[['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']])
heart_minMax=pd.DataFrame(heart_minMax)
heart_minMax.columns=['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']

"""APPENDING ORIGINAL AND TRANSFORMed DATASET"""
heart2=heart[['sex', 'chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope', 'num_major_vessels', 'thalassemia', 'target']]

heart3 = pd.merge(heart2, heart_minMax, left_index=True, right_index=True)
"""building the DNN"""
heart3.columns
x=heart3[['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
       'exercise_induced_angina', 'st_slope', 'num_major_vessels',
       'thalassemia', 'age', 'resting_blood_pressure', 'cholesterol',
       'st_depression', 'max_heart_rate_achieved']]
y=heart3[['target']]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#create model
model = Sequential()
#get number of columns in training data
n_cols = x_train.shape[1]
#3 dense layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
#output layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
##running the nn

model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=100)
eval_model=model.evaluate(x_train, y_train)
eval_model
y_pred=model.predict(x_test)
y_pred =(y_pred>0.5)
## confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
"""r aquare score"""
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

"""plotting some cool graphics"""
import matplotlib.pyplot as plt

# Fit the model
history = model.fit(x_train,y_train,batch_size=10,epochs=100)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''introduce drop out to reduce overfitting'''

model = Sequential()
#get number of columns in training data
n_cols = x_train.shape[1]
#3 dense layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
model.fit(x_train,y_train,batch_size=10,epochs=100)
eval_model=model.evaluate(x_train, y_train)
eval_model
y_pred=model.predict(x_test)
y_pred =(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
history = model.fit(x_train,y_train,batch_size=10,epochs=100)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


