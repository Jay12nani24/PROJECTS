import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('NEW_YIELD.csv')
print(dataset)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# importing algorithm

from sklearn.svm import SVR
svr = SVR()
svr.fit(x_train,y_train)

y_pred = svr.predict(x_test)

area = float(input('enter the area'))
item = float(input('enter the item'))
year = float(input('enter the year'))
rain = float(input('enter the rain_fall'))
pest = float(input('enter the pesticides'))
temp = float(input('enter the tempreature'))

output = svr.predict([[area,item,year,rain,pest,temp]])
print(output)











