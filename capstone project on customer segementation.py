# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:08:51 2022

@author: pc1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from google.colab import drive
drive.mount('/content/drive')
path='/content/drive/Online Retail.xlsx'
dataset=pd.read_excel(path)
dataset.head()
dataset.info()
# Plotting the heatmap for missing values
missing_values=dataset.isnull()
sns.heatmap(missing_values)
# Check null values in customer id column
sum(pd.isnull(dataset['CustomerID']))
# drop nan values
dataset.dropna(subset=['CustomerID'],inplace=True)
# check total entries
len(dataset)
# remove duplicate values
dataset.drop_duplicates(inplace=True)
dataset.info()
# Plotting the heatmap fro missing values
processed_values=dataset.isnull()
plt.figure(figsize=(8,8))
sns.heatmap(processed_values)
dataset.describe()
dataset[dataset['Quantity']<0]
dataset[dataset['Quantity']>0]
dataset.info()
dataset.describe()
# plot sales for  top 10 countries
country_data=dataset.groupby('Country').count().reset_index()
country_data.sort_values('InvoiceNo',ignore_index=True,ascending=False,inplace=True)
fig,axes=plt.subplots(figsize=(16,8))
sns.set_style("darkgrid")
sns.barplot(data=country_data[0:10],x='Country',y='Invoice',ax=axes,linewidth=1,edgecolor='0.2')
axes.set_yticks(range(0,400000,20000))
axes.set_xlabel('Country',size=20)
axes.set_ylabel('Number of transaction',size=20)
axes.set_title('Transactions per country',size=30)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
# get month from datetime object
dataset['Month']=dataset['InvoiceDate'].apply(lambda x:x.month)
dataset.head()
# plot sales for  top 10 countries
month_data=dataset.groupby('Month').count().reset_index()
fig,axes=plt.subplots(figsize=(16,8))
sns.set_style("darkgrid")
sns.barplot(data=month_data,x='Month',y='Invoice',ax=axes,linewidth=1,edgecolor='0.2')
axes.set_yticks(range(0,800000,10000))
axes.set_xlabel('Month',size=20)
axes.set_ylabel('Number of transaction',size=20)
axes.set_title('Transactions per country',size=30)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
# get monetary value or eaxh customer from quantity and unit price
dataset['MonetaryValue']=dataset.apply(lambda x:x['Quantity']*x['UnitPrice'],axis=1)
dataset.head()
customer_data=dataset.groupby('CustomerID').sum().reset_index()
customer_data.head()
# plot monetary values
sns.histplot(customer_data['MonetaryValue'])
customer_data.drop(columns=['Quantity','UnitPrice','Month'],inplace=True)
customer_data['Frequency']=dataset.groupby('CustomerID')['MonetaryValue'].count().values
customer_data.head()
# Plot frequency
sns.histplot(customer_data['Frequency'])
last_date=date(2011,12,10)
dataset['Recency']=dataset['InvoiceDate'].apply(lambda x:(last_date-pd.to_datetime(x).date()).days)
dataset.head()
customer_data['Recency']=dataset.groupby('CustomerID')['Recency'].min().values
customer_data.head()
sns.histplot(customer_data['Recency'])
customer_data.describe()
customer_data['MonetaryValue']=customer_data['MonetaryValue'].apply(lambda x:np.log(x+1))
customer_data['Frequency']=customer_data['Frequency'].apply(lambda x:np.log(x))
customer_data['Recency']=customer_data['Recency'].apply(lambda x:np.log(x))
customer_data.head()
customer_data.describe()
# plot the data distribution after log transform
fig,axis=plt.subplots(nrows=1,ncolumns=3,fogsize=(20,8))
sns.histplot(customer_data['Recency'],ax=axis[0])
sns.histplot(customer_data['Frequency'],ax=axis[1])
sns.histplot(customer_data['MontearyValue'],ax=axis[2])
fig.subtitle("Data distribution after log transform",size=25)
# plot data points in 3D space
fig=plt.figure(figsize=(8,8))
ax=Axes3D(fig)
x=customer_data['Recency']
y=customer_data['Frequency']
z=customer_data['MontearyValue']
ax.scatter(x,y,z,marker='.')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('MonetaryValue')
plt.title("Data Visualization",size=25)
customer_data.head()
data=customer_data.drop(columns=['CustomerID'])
data.head()
sse={}
for k in range(1,21):
   kmeans=KMeans(n_clusters=k,random_state=1)
   kmeans.fit(data)
   sse[k]=kmeans.inertia_
# the elbow plot
plt.figure(figsize=(12,8))
plt.title('The Elbow Mehod',size=25)
plt.xlabel('k',size=20)
plt.ylabel('Sum of squared errors',size=20)
sns.pointplot(x=list(sse.keys()),y=list(sse.values()))
cluster_list=[2,3,4,5,6,7,8]
for n_clusters in cluster_list:
    clusterer=KMeans(n_clusters=n_clusters,random_state=1)
    cluster_labels=clusterer.fit_predict(data)
    silhouette_avg=silhouette_score(data,cluster_labels)
    print("for n_clusters=",n_clusters,"The average sillhouettes_score is :",silhouette_avg)
    
    