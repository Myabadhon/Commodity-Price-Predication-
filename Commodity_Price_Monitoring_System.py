# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 06:48:23 2020

@author: myaba
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Import DataFrame/Dataset
df = pd.read_csv('Commodity_Price.csv')

#ensure that date columns are parsed correctly as Datetime you must implicitly add them
dateCols = ['Date']
pd.read_csv("Commodity_Price.csv", parse_dates=dateCols)
#In order to convert a column stored as a object/string into a DataFrame
df.Date=pd.to_datetime(df.Date, utc= False)
#Verify columns containing dates
df.dtypes


#Get The Number Of Column
df.shape

#Visualize The Atash Rice Price
plt.figure(figsize = (16,4))
plt.title('Atash Price History')
plt.plot(df['Atash Rice'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Atash Rice Price (Taka)', fontsize = 18)
plt.show()

#Create a ne Dataframe With Only Atash Rice
data = df.filter(['Atash Rice'])
#Convert The dataframe into a numpy array
dataset = data.values
#Get The number Of  Rows to Train The Model
training_data_len = math.ceil( len(dataset) * .8 )
training_data_len

#Scale The Data(It's Just For Good Practice)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


#Create teh training Dataset
#create The Scaled training Dataset
train_data = scaled_data[0:training_data_len, :]
#Split The Data Into X train And Y train data Sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()
        
#Convert The X train And Y train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data in Theree Dimensional array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#Or x_train = np.reshape(x_train, (233,60, 1))
x_train.shape

#Build a LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compile The Model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Train The Model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#Create The Testing Data Set
#Create a new array containing scaled Values from index 234 to 366
test_data = scaled_data[training_data_len - 60: , :]
#Create The Test DataSet x test And Y_test
x_test =[]
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60 : i, 0])

#Convvert The test data into A numpy Array
x_test = np.array(x_test)

#Reshape The Test Data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get The Model Pridected Price Values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get The Root Mean Squared Error (RMSE)
rmse = np.sqrt( np.mean( predictions - y_test)**2 )
rmse

#Plot The Data
train = data[ : training_data_len]
valid = data[training_data_len : ]
valid['predictions'] = predictions

#Visualize The Data 
plt.figure(figsize =(16,4))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Atash Rice Price (Taka)', fontsize = 18)
plt.plot(train['Atash Rice'])
plt.plot(valid[['Atash Rice', 'predictions']])
plt.legend(['Train', 'Val', 'predictions'], loc = 'lower right')
plt.show()

valid




















