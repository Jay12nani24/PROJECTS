import pandas as pd
import numpy as np

dataset = pd.read_csv("C:\\Users\\Anbazhagan R\\Desktop\\MANGOYIELD PREDICTION\\CROP YIELD PREDICTION\\YIELD.csv")
print(dataset)


Area=dataset.iloc[:,:-6].values
print(Area)
Item=dataset.iloc[:,-6:-5].values
print(Item)

print("LABEL ENCODER PREPROCESSING")
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
Area= labelencoder.fit_transform(Area.ravel())
print(Area)
Item= labelencoder.fit_transform(Item.ravel())
print(Item)


dataset = pd.read_csv('C:\\Users\\Anbazhagan R\\Desktop\\MANGOYIELD PREDICTION\\CROP YIELD PREDICTION\\NEW_YIELD.csv')
print(dataset)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print("SPLITTING DATASET TRAIN AND TESR")
from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.20 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)

print(" K NEIGBOR CLASSIFICATION")
from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(x_train,y_train)
print("TRAINING ACCURACY")
print('Support Vector Machine Training Accuracy:', regressor.score(x_train, y_train))


a1=int(input("ENTER THE AREA= "))
b1=int(input("ENTER THE ITEM= "))
c1=int(input("ENTER THE YEAR= "))
d1=int(input("ENTER THE AVERAGE RAIN FALL PER YEAR= "))
e1=int(input("ENTER THE PESTICIDES TONNES= "))
f1=float(input("ENTER THE AVERAGE TEMPERATURE= "))

a= knn.predict([[a1,b1,c1,d1,e1,f1]])
print('Predicted new output value: %s' % (a))






