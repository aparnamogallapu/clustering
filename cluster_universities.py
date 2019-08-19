# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:16:57 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset1=pd.read_csv("F:\\AI\\Datasets\\Universities.csv")
x=dataset1.iloc[:,[1,5]].values

#elbow method:
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('The Number Of clusters')
plt.ylabel('wcss')
plt.show()

#fitting dataset
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)


%matplotlib
#visulization:
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],c='red',s=100,label='sensible')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='green',s=100,label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='blue',s=100,label='target')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],c='yellow',s=100,label='careless')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',s=300,label='centroids')
plt.title('Clusters of Universities')
plt.xlabel('Sat Score')
plt.ylabel('Expenses')
plt.legend()
plt.show()