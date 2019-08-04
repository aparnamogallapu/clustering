# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:25:22 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset2=pd.read_csv("C:\\Users\\HP\\Desktop\\u_datasets\\K_Means\\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#elow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS')
plt.show()


#fit kmeans dataset
kmeans=KMeans(n_clusters = 5,init = 'k-means++', random_state = 0)
y_kmenas=kmeans.fit_predict(X)

#visualization
plt.scatter(X[y_kmenas==0,0],X[y_kmenas==0,1],s=100,c='red',label='carefull')
plt.scatter(X[y_kmenas==1,0],X[y_kmenas==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_kmenas==2,0],X[y_kmenas==2,1],s=100,c='green',label='target')
plt.scatter(X[y_kmenas==3,0],X[y_kmenas==3,1],s=100,c='yellow',label='careless')
plt.scatter(X[y_kmenas==4,0],X[y_kmenas==4,1],s=100,c='black',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroids')
plt.title('clusters of customers')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()

