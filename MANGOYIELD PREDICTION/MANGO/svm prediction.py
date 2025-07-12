#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


data = pd.read_csv('mango_dataset.csv')

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)
X_train = training_set.iloc[:,:-1].values
print(X_train)
Y_train = training_set.iloc[:,-1].values
print(Y_train)
X_test = test_set.iloc[:,:-1].values
Y_test = test_set.iloc[:,-1].values

from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
#print("support vector accuracy",r2_score(Y_test, y_pred))

score_train = mean_squared_error(Y_test, y_pred)
print("accuracy score of support vector machine is = ",score_train)

