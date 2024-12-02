import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
print(dataset.head(5))
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
p_r=PolynomialFeatures(degree=4)
x_poly=p_r.fit_transform(x)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
LinearRegression()

y_pred=lin_reg.predict(x_poly)
df=pd.DataFrame({'Real Values':y,'predicted values':y_pred})
print(df)

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid)))
plt.scatter(x,y,color="yellow")
plt.scatter(x,y_pred,color="red")
plt.plot(x_grid,lin_reg.predict(p_r.fit_transform(x_grid)),color='black')
plt.title('polynomial regression')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

