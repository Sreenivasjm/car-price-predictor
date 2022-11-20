#importing required modules 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle as pkl

# reading the data from dataset
df=pd.read_csv('cleaned_car.csv')
df.info()

df['fuel_type'].unique()

#preprocessing the data

df=df[df['year'].str.isnumeric()]
df['name']=df['name'].str.split(' ').str.slice(0,3).str.join(' ')
df['year'].astype(int)

df=df[df['Price']!='Ask For Price']

df['Price']=df['Price'].str.replace(',','').astype(int)

df['kms_driven']=df['kms_driven'].str.split(' ').str.slice(0,1).str.join(' ').str.replace(',','')

df=df[df['kms_driven']!='Petrol']

df['kms_driven']=df['kms_driven'].astype(int)

df=df[~df['fuel_type'].isna()]

df=df[df['Price']<6e6]
df.reset_index(drop=True)

#storing the preprocessed data
df.to_csv('cleaned_car.csv')

#crating the features data
X=df.drop(columns='Price')

#creating the labels data
y=df['Price']


#splitting the data in training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.1)

#creating the object model
lr=LinearRegression()

#changing the ordial data into binary using onehotencoder
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
#dd=pd.read_csv('https://bit.ly/kaggletrain')


#column transformer
col_transform=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


#joining the pipe line and linear regression model
pipe=make_pipeline(col_transform,lr)

#fitting the training set into the model
pipe.fit(X_train,Y_train)

#predicting the model
y_pred=pipe.predict(X_test)

#accuracy of the model
score=r2_score(Y_test,y_pred)


#testing the best score in the model
score=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(col_transform,lr)
    pipe.fit(X_train,Y_train)
    y_pred=pipe.predict(X_test)
    score.append(r2_score(Y_test,y_pred))

#predicting the car using the data
pipe.predict(pd.DataFrame([['Skoda Yeti Ambition','Skoda',2012,6800,'Diesel']],columns=['name','company','year','kms_driven','fuel_type']))
