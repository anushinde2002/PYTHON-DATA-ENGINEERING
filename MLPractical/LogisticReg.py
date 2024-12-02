import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()

print(list(iris.keys()))

#print data
print(iris['data'])
print(iris['target'])

#print data and corresponding targets of that data
print(iris['DESC'])
print(iris['data'].shape)
x=iris["data"][:,3]
print(x)

y=(iris["target"]==2)
print(y)
y=(iris["target"]==2).astype(np.int64)
print(y)

clf=LogisticRegression()
clf.fit(x,y)
example=clf.predict(([[1,6]]))
print(example)

example=clf.predict(([[2.6]]))
print(example)

x_new=np.linspace(0,3,1000).reshape(-1,1)
print(x_new)
y_prob=clg.predict_proba(x_new)
print(y_prob)
plt.plot(x_new,y_prob[:,1],"g-",label='verginica')
plt.show()
