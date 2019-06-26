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

# 
