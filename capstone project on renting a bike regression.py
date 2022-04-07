# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:11:49 2022

@author: pc1
"""
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import datetime as dt
style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,VotingRegressor,StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from catboost import CatBoostRegressor
from scipy import stats
from google.colab import drive
drive.mount('/content/drive')
directory='/content/drive//MyDrive/SeoulBikeData.csv'
df=pd.read_csv(directory,encoding='latin')
print("Shape of the resources dataFrame is :",df.shape)
df.head(24)
df.tail()
df.dtypes
df.describe()
df.info()
df.columns
len(df.Date.unique())
df.Seasons.value_counts()
df.Holiday.value_counts()
df['Functioning Day'].value_counts()
plt.figure(figsize=(17,5))
sns.heatmap(df.isnull(),cbar=True,yticklabels=False)
plt.xlabel("Column_Name",size=14,weight="bold")
plt.title("Places of missing values in column",fontweight="bold",size=17)
plt.show()
# Unique values in each columns
unique_df=pd.DataFrame()
unique_df['Features']=df.columns
unique=[]
for i in df.columns:
    unique.append(df[i].nunique())
unique_df['Uniques']=unique
f,ax=plt.subplots(1,1,figsize=(15,7))
splot=sns.barplot(x=unique_df['Features'],y=unique_df['Uniques'],alpha=0.8)
for p in splot.patches:
    splot.annotate(format(p.get_height(),'.0f'),(p.get_x()+p.get_width()/2.,p.get_height()),ha='center',
                   va='center',xytext=(0,9),textcoords='offset points')
plt.title('Bar plot for number of unique values in each column',weight='bold',size=15)
plt.ylabel('#Unique Values',size=12,weight='bold')
plt.xlabel('Features',size=12,weight='bold')
plt.xticks(rotation=90)
plt.show()
import matplotlib as mpl
mpl.rc('font',size=15)
sns.distplot(df['Rented Bike Count']);
fig=plt.figure(figsize=(15,20))
ax=fig.gca()
df.hist(ax=ax)
df.plot.hist(by=None,bins=10)
df.rename(columns={'Rented Bike Count':'RentedBikeCount','Wind Speed (m/s)':'WindSpeed','Visibility (10m)':'Visibility',
                   'Dew Point Temperature (`c)':'DewPointTemperature','Solar Radiation (MJ/m2)':'SolarRadiation','Rainfall (mm)':'Rainfall','Snowfall (cm)':'Snowfall',
                   'Functioning Day':'FunctioningDay','Temperature (`C)':'Temperature','Humidity(%)':'Humidity'},inplace=True)
sns.boxplot(df['RentedBikeCount'])
df.head(1)
df['Date']=pd.to_datetime(df['Date'])
# Label encoding the data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Holiday']=le.fit_transform(df['Holiday'])
df['FunctioningDay']=le.fit_transform(df['FunctioningDay'])
df['Seasons']=le.fit_transform(df['Seasons'])
df.head(1)
df['Seasons'].value_counts()
df['FunctioningDay'].value_counts()
df['Holiday'].value_counts()
plt.figure(figsize=(20,10))
sns.boxplot(x="Date",y="RentedBikeCount",data=df)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap='Spectral_r')
df.info()
# Importing library for VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif=pd.DataFrame()
    vif["variables"]=X.columns
    vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    return (vif)
df['Date']=df['Date'].values.astype(float)
X=df.iloc[:,:-1]
calc_vif(X)
df.drop(['Date'],axis=1,inplace=True)
# Importing library for VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
     vif=pd.DataFrame()
     vif["variables"]=X.columns
     vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
     return (vif)
X=df.iloc[:,:-1]
calc_vif(X)
df.drop(['Temperature'],axis=1,inplace=True)
# Importing library for VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
     vif=pd.DataFrame()
     vif["variables"]=X.columns
     vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
     return (vif)
X=df.iloc[:,:-1]
calc_vif(X)
df.head(1)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap='Spectral_r')
df.drop(['Humidity'],axis=1,inplace=True)
# Importing library for VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
     vif=pd.DataFrame()
     vif["variables"]=X.columns
     vif["VIF"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
     return (vif)
X=df.iloc[:,:-1]
calc_vif(X)
df.head(1)
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('FunctioningDay')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['FunctioningDay'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('Holiday')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['Holiday'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('Seasons')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['Seasons'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('Rainfall')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['Rainfall'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('SolarRadiation')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['SolarRadiation'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('DewPointTemperature')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['DewPointTemperature'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('Visibility')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['Visibility'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('WindSpeed')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['WindSpeed'],y=df['RentedBikeCount'])
plt.figure(figsize=(5,5))
plt.title('Linear Regression')
plt.xlabel('Hour')
plt.ylabel('RentedBikeCount')
sns.regplot(x=df['Hour'],y=df['RentedBikeCount'])
# Skewness of data
import missingno as mso
mso.matrix(df,figsize=(12,5))
# Linear Regression
X=df[['Hour','WindSpeed','Visibility','DewPointTemperature','SolarRadiation','Rainfall','Snowfall','Seasons','Holiday','FunctioningDay']]
y=np.sqrt(df['RentedBikeCount'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X.shape)
print(y.shape)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
regressor.coef_
# Predicting the Test set results
y_pred=regressor.predict(X_test)
import math
math.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
y_train_pred=regressor.predict(X_train)
y_test_pred=regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error
def print_metrices(actual,predicted):
    print('MSE is {}'.format(mean_squared_error(actual,predicted)))
    print('RMSE is {}'.format(math.sqrt(mean_squared_error(actual,predicted))))
    print('RMSE is {}'.format(r2_score(actual,predicted)))
    print('MAE is {}'.format(mean_absolute_error(actual,predicted)))
    print('MAPE is {}'.format(np.mean(np.abs((actual-predicted)/actual))*100))
print_metrices(y_train, y_train_pred)
print_metrices(y_test, y_test_pred)
residuals_train=y_train-y_train_pred
residuals_test=y_test-y_test_pred
plt.scatter(y_train_pred,residuals_train,c='red')
plt.title('Scatter Plot between residuals and actual profits')
plt.xlabel('Predicted Profit')
plt.ylabel('Residual')
plt.show()
round((np.mean(residuals_train)))
def adjusted_r2(n,k,actual,predicted):
    return 1-(((n-1)/(n-k-1))*(1-r2_score(actual,predicted)))
adjusted_r2(len(y_train), len(X), y_train, y_train_pred)
# Polynomial Linear Regression
from sklearn.preprocessing  import PolynomialFeatures
def create_polynomial_regression_model(degree):
    "Creates a polynomial regression model of the given degree"
    poly_features=PolynomialFeatures(degree=degree)
    X_train_poly=poly_features.fit_transform(X_train)
    poly_model=LinearRegression()
    poly_model.fit(X_train_poly,y_train)
    y_train_predicted=poly_model.predict(X_train_poly)
    y_test_predict=poly_model.predict(poly_features.fit_transform(X_test))
    rmse_train=np.sqrt(mean_squared_error(y_train,y_train_predicted))
    r2_train=r2_score(y_train,y_train_predicted)
    rmse_test=np.sqrt(mean_squared_error(y_test,y_test_predict))
    r2_test=r2_score(y_test,y_test_predict)
    print("The model performance for the training set")
    print("------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
    print("\n")
    print("The model performance for the training set")
    print("------------------------------------------")
    print("RMSE of training set is {}".format(rmse_test))
    print("R2 score of training set is {}".format(r2_test))
create_polynomial_regression_model(3)
create_polynomial_regression_model(5)
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from skearn.datasets import make_regression
X,y=make_regression(n_features=4,n_n_informative=2,
                    random_state=0,shuffle=False)
regr=RandomForestRegressor(max_depth=10,random_state=0)
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
import math
math.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
y_train_pred=regr.predict(X_train)
y_test_pred=regr.predict(X_test)
from sklearn.metrics import mean_absolute_error
def print_metrices(actual,predicted):
    print('MSE is {}'.format(mean_squared_error(actual,predicted)))
    print('RMSE is {}'.format(math.sqrt(mean_squared_error(actual,predicted))))
    print('RMSE is {}'.format(r2_score(actual,predicted)))
    print('MAE is {}'.format(mean_absolute_error(actual,predicted)))
    print('MAPE is {}'.format(np.mean(np.abs((actual-predicted)/actual))*100))
print_metrices(y_train, y_train_pred)
print_metrices(y_test, y_test_pred)
# CatBoost Regressor
from catboost import CatBoostRegressor
import timeit
from skearn.datasets import make_regression
model=CatBoostRegressor(
    iterations=100,
    learning_rate=0.03
    )
model.fit(
    X_train,y_train,
    eval_set=(X_test,y_test),
    verbose=10
    );
def train_on_cpu():
    model=CatBoostRegressor(
        iterations=100,
        learning_rate=0.03
        )
    model.fit(
        X_train,y_train,
        eval_set=(X_test,y_test),
        verbose=10
        );
    cpu_time=timeit.timeit('train_on_cpu()',
                           setup="from_main_import train_on_cpu",
                           number=1)
    print('Time to fit model on CPU: {} sec'.format(int(cpu_time)))
y_pred=model.predict(X_test)
import math
math.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
y_train_pred=regr.predict(X_train)
y_test_pred=regr.predict(X_test)
from sklearn.metrics import mean_absolute_error
def print_metrices(actual,predicted):
    print('MSE is {}'.format(mean_squared_error(actual,predicted)))
    print('RMSE is {}'.format(math.sqrt(mean_squared_error(actual,predicted))))
    print('RMSE is {}'.format(r2_score(actual,predicted)))
    print('MAE is {}'.format(mean_absolute_error(actual,predicted)))
    print('MAPE is {}'.format(np.mean(np.abs((actual-predicted)/actual))*100))
print_metrices(y_train, y_train_pred)
print_metrices(y_test, y_test_pred)   
# Modelling
df['RentedBikeCount']=df['RentedBikeCount'].map(np.log1p)
df.head()
sns.boxplot(df['RentedBikeCount'])
X=df.drop(["RentedBikeCount"],axis=1)
y=df.RentedBikeCount
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
ridge= Ridge()
ridge_params={'alpha':(0.001,0.005,0.01,0.05,0.1,0.5,1)}
grid=GridSearchCV(ridge,ridge_params,scoring='r2')
grid.fit(X_train,y_train)
ridge=grid.best_estimator_
svr=SVR()
svr_params={'kernel':('linear','poly','rbf'),
            'C':(0.001,0.005,0.01,0.05,0.1,0.5,1)}
grid=GridSearchCV(svr,svr_params,scoring='r2')
grid.fit(X_train,y_train)
svr=grid.best_estimator_
kneighb=KNeighborsRegressor()
kneighb_params={'n_neighbors':(range(2,15))}
grid=GridSearchCV(kneighb,kneighb_params,scoring='r2')
grid.fit(X_train,y_train)
kneighb=grid.best_estimator_
rf=RandomForestRegressor().fit(X_train,y_train)
voting_estimators=[('svr',svr),('rf',rf),('kneighb',kneighb),('ridge',ridge)]
voting=VotingRegressor(voting_estimators).fit(X_train,y_train)
names=['svr','kneighb','rf','ridge','voting']
scores=[]
for counter,i in enumerate([svr,kneighb,rf,ridge,voting]):
    scores.append(mean_squared_error(y_test,i.predict(X_test)))
tmp=pd.DataFrame(scores,names).T
plt.figure(figsize=(15,5))
plt.bar(names,scores,align='edge')
tmp.head(5)
xgb=XGBRegressor(n_estimators=700,
                 max_depth=7,
                 learning_rate=0.05)
xgb.fit(X_train,y_train)
lgb=LGBMRegressor(n_estimators=600,
                 max_depth=6,
                 learning_rate=0.1)
lgb.fit(X_train,y_train)
ctb=CatBoostRegressor(n_estimators=500,
                 max_depth=5,
                 learning_rate=0.1,verbose=0)
ctb.fit(X_train,y_train)
gb_ensemble=[('xgb',xgb),('lgb',lgb),('ctb',ctb)]
voting_gb=VotingRegressor(gb_ensemble).fit(X_train,y_train)
gb_ensemble=[('xgb',xgb),('lgb',lgb),('ctb',ctb)]
voting_gb=VotingRegressor(gb_ensemble).fit(X_train,y_train)
names_gb=['xgb','lgb','ctb','gb ensemble']
scores_gb=[]
for counter,i in enumerate([xgb,lgb,ctb,voting_gb]):
    scores_gb.append(mean_squared_error(y_test,i.predict(X_test)))
tmp=pd.DataFrame(scores_gb,names_gb).T
tmp.head(5)
