import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dict={'first score':[100,20,np.nan,50],
      'second score':[20,30,40,np.nan],
      'third score':[np.nan,30,40,50]}
df=pd.DataFrame(dict)
print(df)
x=df.isnull()
print(x)
y=df.notnull()
print(y)
z=df.fillna(0)
print(z)
s=df.fillna(method='pad')
print(s)
a=df.fillna(method='bfill')
print(a)
b=df.replace(to_replace=np.nan,value=-99)
print(b)
c=df.dropna()
print(c)
d=df.dropna(axis=1)
print(d)
new_data=df.dropna(axis=0)
