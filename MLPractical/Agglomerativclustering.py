import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
x=pd.read_csv("CC_GENRAL.csv")
x=x.drop('CUST_ID',axis=1)
x.fillna(method='ffill',inplace=True)
scaler=StandardScaler
x_scaled=scaler.fit_transform(x)
x_normalized=normalize(x_scaled)

x_normalized=pd.DataFrame(x_normalized)
pca=PCA(n_components=2)
x_principal=pca.fit_transform(x_normalized)
x_principal=pd.DataFrame(x_principal)
x_principal.columns=['p1','p2']
plt.figure(figsize=(8,8))
plt.title('visualizing the data')
dendrogram=shc.dendrogram((shc.linkage(x_principal,method='word')))
plt.show()

