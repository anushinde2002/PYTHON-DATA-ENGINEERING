import numpy as np
from sklearn.naive_bayes import GaussianNB
x_train=np.array([1,2],[3,4],[5,6],[7,8])
y_train=np.array([1,1,2,2])
x_test=np.array([9,10],[11,12],[13,14],[15,16])
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print("accuracy",classifier.score(x_test,y_pred))
