# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:27:26 2022

@author: pc1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
data=pd.read_csv("Corona_NLP_train.csv",enacoding='latin1')
df=pd.DataFrame(data)
df.head()
plt.figure(figsize=(10,5))
sns.countplot(x='Sentiment',data=df,order=['Extremely Negative','Negative','Neutral','Positive','Extreme Positive'],)
df.info()
reg=re.compile("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z t])|(w+://S+)")
tweet=[]
for i in df["OriginalTweet"]:
    tweet.append(reg.sub(" ",i))
df=pd.concat([df,pd.DataFrame(tweet,columns=["CleanedTweet"])],axis=1,sort=False)
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words=set(stopwords.words('english')) # make a set of stopwords
vectoriser=TfidfVectorizer(stop_words=None)
X_train=vectoriser.fit_transform(df["CleanedTweet"])
# Encoding the classes in numerical values
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
y_train=encoder.fit_transform(df['Sentiment'])
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train) 
# importing the Test Dataset for prediction and testing purposes
test_data=pd.read_csv("Corona_NLP_train.csv",enacoding='latin1')
test_df=pd.DataFrame(test_data)
test_df.head()
reg1=re.compile("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z t])|(w+://S+)")
tweet=[]
for i in test_df["OriginalTweet"]:
    tweet.append(reg1.sub(" ",i))
test_df=pd.concat([test_df,pd.DataFrame(tweet,columns=["CleanedTweet"])],axis=1,sort=False)
test_df.head()
X_test=vectoriser.transform(test_df["CleanedTweet"])
y_test=encoder.fit_transform(df["Sentiment"])
# Prediction
y_pred=classifier.predict(X_test)
pred_df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
pred_df.head()
from sklearn import metrices
# Generate the roc curve using scikit-learn.
fpr,tpr,thresholds=metrices.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()
# Measure the area under the curve. The closer to 1 ,the "better" the predictions.
print("AUC of the predictions : {0}".format(metrices.auc(fpr,tpr)))


