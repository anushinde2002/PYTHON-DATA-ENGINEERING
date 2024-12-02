import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iris=pd.read_csv("C:\Users\Admin\Downloads\Iris.csv")
print(iris.head(20))
plt.plot(iris.id,iris['sepallengthcm'],"r--")
plt.show
iris.plot(kind="scatter",x="sepalLengthCm",y="petalLengthCm")
plt.show()
