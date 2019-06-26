# Classification-example-using-Keras
Predicting Heart disease using Age, Sex, blood Sugar, blood pressure, cholesterol etc. using Neural Network

Hi All,

I would like you to take you through the one basic example of building neural network using Keras. It uses a dataset of heart diseases which can be downloaded on Kaggle as well. 
# the dataset
We have the following columns.

1.age in years
2.sex-(1 = male; 0 = female)
3.cpchest- pain type
4.trestbpsresting blood pressure (in mm Hg on admission to the hospital)
5.cholserum cholestoral in mg/dl
6.fbs(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7.restecgresting electrocardiographic results
8.thalachmaximum heart rate achieved
9.exangexercise induced angina (1 = yes; 0 = no)
10.oldpeakST depression induced by exercise relative to rest
11.slopethe slope of the peak exercise ST segment
12.canumber of major vessels (0-3) colored by flourosopy
13.thal3 = normal; 6 = fixed defect; 7 = reversable defect
14.target1 or 0 (heart disease yes or no)
# Renaming the variables to better understand 

heart.columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
 # Generic correlation to find what impacts the heart attack
 Just checking what variables have strong correlation.Please only concentrate on last row of correlation matrix.
 
heart.corr()
 # Visualization by classes to better understand the factors
 We will use seaborn library to get beuatiful scatter plot matrix for numeric variables.
 
 heart_num=heart[['age','resting_blood_pressure','cholesterol','st_depression','target','max_heart_rate_achieved']]
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(heart_num, hue="target")

# min max scaling(converting evrything between 0 and 1 ..basic data prep for nn
We will also do min max scaling to pre process our data so that it fits our model.

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler().fit(heart[['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']])
heart_minMax=minmax_scale.transform(heart[['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']])
heart_minMax=pd.DataFrame(heart_minMax)
heart_minMax.columns=['age','resting_blood_pressure','cholesterol','st_depression','max_heart_rate_achieved']

 # APPENDING ORIGINAL AND TRANSFORMed DATASET
 
 heart2=heart[['sex', 'chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope', 'num_major_vessels', 'thalassemia', 'target']]

heart3 = pd.merge(heart2, heart_minMax, left_index=True, right_index=True)
"""building the DNN"""
heart3.columns
x=heart3[['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
       'exercise_induced_angina', 'st_slope', 'num_major_vessels',
       'thalassemia', 'age', 'resting_blood_pressure', 'cholesterol',
       'st_depression','max_heart_rate_achieved']]
y=heart3[['target']]
       
       
 # lets build a basic DNN using keras
  well all thanks to this beautiful book:
  
  https://www.amazon.in/Deep-Learning-Python-Francois-Chollet/dp/1617294438
  
  We will be building a sequential NN with 3 deep layers , as per my experience for binary classification the best activation is softmax the best optimizer is Adam and for deep layers the best activation is RELU. I have tried RMS Prop as well but it yielded inferior results, binary cross entropy is loss function.
  
  I am measuring accurcay over the epochs. You can try out the different combinations of deep layers optimizer and activations and see the results.
  
  from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
## create model
model = Sequential()
## get number of columns in training data
n_cols = x_train.shape[1]
## 3 dense layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
## output layer
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
## running the nn

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
## r aquare score"""
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
  
# plotting some cool graphics the accuracy and loss over the epochs

mport matplotlib.pyplot as plt

## Fit the model
history = model.fit(x_train,y_train,batch_size=10,epochs=100)
## list all data in history
print(history.history.keys())
## summarize history for accuracy
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

# the droput - for reducing the overfitting 

Please read the theory of drop out from here:

https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5

model = Sequential()
## get number of columns in training data
n_cols = x_train.shape[1]
## 3 dense layers
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
## list all data in history
print(history.history.keys())
## summarize history for accuracy
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


