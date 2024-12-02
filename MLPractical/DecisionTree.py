import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
df=pd.read_csv("playtennis.csv")
print(df)
x=df[['outlook','temperature','humidity','windy']]
y=df['play']
le=LabelEncoder()
x['outlook']=le.fit_transform(x['outlook'])
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x,y)
tree.plot_tree(clf)
