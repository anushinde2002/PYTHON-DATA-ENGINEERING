import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
data=pd.read_csv(r'kc_house_data.csv')
data.head(5)
print(data.shape)
f=['price','bedrooms','bathrooms','sqft_living','condition','sqft_above','sqft_basement','yr_built','yr_renovated']
data=data[f]

print(data.shape)
data=data.dropna()
print(data.shape)
data.describe()
x=data[f[1:]]
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.coef_)
y_test_predict=lr.predict(x_test)
print(y_test_predict.shape)
g=plt.plot((y_test-y_test_predict),marker='0',linestyle="")
plt.show()
