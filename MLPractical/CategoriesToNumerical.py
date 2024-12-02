import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# df=pd.read_csv("iris.csv")
# print(df)
S={'Id':[1,2,3,4,5,6,7,8,9,10],
   'Name':['anita','ritika','sonal','sakshi','sayali','vedika','harshada','aniket'],
   'marks':[95,69,97,83,89,49,96,93]}
S=pd.read_csv(S)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
label=le.fit_transform(df['play'])
print(label)
df.drop('play', axis=1,inplace=True)
df['play']=label
print(df)
