import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
x=np.array([[1,2],[5,8],[8,8],[1,0.6]])
y=[1,0,1,0]
clf=svm.svc(kernel='linear'(1.0).fit(x,y))
print(clf.predict[[0.56,0.76]])
w=clf=coef_[0]
print(w)
a=-w[0]/w[1]
xx=np.linspace(0,12)
yy=a*xx-clf.intercept_[0]/w[1]
plt.plot(xx,yy,'k-',label='non weighted div')
plt.scatter(x[:p],x[:,1],c=y)
plt.legend()
plt.show()


