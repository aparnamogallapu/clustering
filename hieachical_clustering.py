# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:55:42 2019

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\u_datasets\\K_Means\\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

##dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Custermers')
plt.ylabel('Euclidean distance')
plt.show()

##fitting hierarcal clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#visulization:
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],c='red',s=100,label='carefull')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],c='green',s=100,label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],c='blue',s=100,label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],c='yellow',s=100,label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='black',label='sensible')
plt.title('clusters of customers')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()
