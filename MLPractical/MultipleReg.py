import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("housingdata.csv")
x=df[['price']]
y=df[['lotsize']]

reg=LinearRegression().fit(x,y)
y_pred=reg.predict(x)
print(y_pred)

score=re_score(ymy_pred)
print(score)
plt.plot(x,y)
plt.show()
