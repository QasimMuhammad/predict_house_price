#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:52:53 2017

@author: qasim
"""


# Refer to kaggle dataset https://www.kaggle.com/hm-land-registry/uk-housing-prices-paid

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import time
import wget
url 'https://www.kaggle.com/harlfoxem/housesalesprediction/downloads/housesalesprediction.zip'
house_prices = wget.download(url)
house_prices

zip_ref = zipfile.ZipFile("housesalesprediction.zip",'r')
zip_ref.extractall()
zip_ref.close()


data = pd.read_csv('price_paid_records.csv', sep=',')

        
data['Date of Transfer'] = pd.to_datetime(data['Date of Transfer'])
data['Month_code'] = data['Date of Transfer'].dt.month
 
data['Property_Type_code']=pd.Categorical(data["Property Type"]).codes
data['old_new_code']=pd.Categorical(data["Old/New"]).codes
data['duration_code']=pd.Categorical(data["Duration"]).codes
data['Cat_code']=pd.Categorical(data["PPDCategory Type"]).codes

data= pd.get_dummies(data,columns=["Month_code","Property_Type_code","old_new_code","duration_code","Cat_code"],prefix=["Month","type","old_new","duration","cat"])

data = data.drop(["Transaction unique identifier","Date of Transfer","Property Type","Old/New","Duration","Town/City","District","County","PPDCategory Type","Record Status - monthly file only"],axis=1)
data.shape
X=data.values
y = data['Price']
data = data.drop(["Price"],axis = 1)
data.close()
X_train, X_val, y_train, y_val =  train_test_split(X,y,test_size=0.2,random_state = 0)

regr = linear_model.LinearRegression()
start_time =time.time()
regr.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
y_train_pred=regr.predict(X_train)
print("Mean squared error: %2e" % np.mean((regr.predict(X_train) - y_train) ** 2))
print("Mean squared error with validation set: %2e" % np.mean((regr.predict(X_val) - y_val) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %e' % regr.score(X_train, y_train))
print('Variance score with validation set: %e' % regr.score(X_val, y_val))