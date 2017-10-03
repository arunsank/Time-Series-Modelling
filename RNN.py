# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:43:06 2017

@author: Arun
"""
import os
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#IMPORT DATA 
train_data = pd.read_csv('Google_Stock_Price_Train.csv')
train_ar = train_data.iloc[:,1:2].values

test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_ar = test_data.iloc[:,1:2].values



scal = MinMaxScaler()

train_ar= scal.fit_transform(train_ar)

test_ar = scal.transform(test_ar)


xtrain= train_ar[0:1257]
ytrain= train_ar[1:1258]

"""
Create time step - time step is the Differece between the times

Model 1: 
The time step in this case is 1

Hint
The reshape is to change the format to a 3 dimensional array to fit our Keras regressor
#The three dimensional array contains the observations , the time step and the dimension of
features respectively as the 3 diemnsional input format for keras. Refer Keras for more info
"""

xtrain = np.reshape(xtrain,(len(xtrain),1,1))
xtest = np.reshape(test_ar,(len(test_ar),1,1))

#Create Model

regressor= Sequential()
regressor.add(LSTM(units=4, activation ='sigmoid',input_shape= (None,1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer ='rmsprop', loss= 'mean_squared_error')
regressor.fit(xtrain,ytrain,batch_size= 32, epochs=200)

#predictions
predicted_vals = regressor.predict(xtest)
predicted_vals = scal.inverse_transform(predicted_vals)


actual_price = test_data.iloc[:,1:2].values



plt.plot(actual_price, color= 'green', label= 'actual stock price')
plt.plot(predicted_vals, color= 'blue', label= 'predicted stock price')
plt.title('Stock price analysis')
plt.xlabel('Time(days)')
plt.ylabel('Price')
plt.legend()
plt.show()


#Evaluate RNN

rmse = math.sqrt(mean_squared_error(actual_price,predicted_vals))

print ("The root mean squared value of our model= {}".format(rmse) )

"""
Output:
    
The root mean squared value of our model= 2.8683133605467805

"""



"""

Make four models:
Model 1 - 20 timesteps & 1 LSTM layer
Model 2 - 20 timesteps & 4 LSTM layers
Model 3 - 60 timesteps & 1 LSTM layer
Model 4 - 60 timesteps & 4 STM layers

""""

#Model 2 - 20 timesteps & 1 LSTM layer



xtrain=[] #array with t-20
ytrain=[] #array with t


for index in range(20,len(train_ar)):
    xtrain.append(train_ar[index-20:index,0])
    ytrain.append(train_ar[index,0])
    

xtrain= np.array(xtrain)
ytrain = np.array(ytrain)
    

xtrain = np.reshape(xtrain,(len(xtrain),xtrain.shape[1],1))

#Regressor
regressor= Sequential()
regressor.add(LSTM(units=4, activation ='sigmoid',input_shape= (None,1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer ='rmsprop', loss= 'mean_squared_error')
regressor.fit(xtrain,ytrain,batch_size= 32, epochs=200)


test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_ar = test_data.iloc[:,1:2].values
test_ar = scal.transform(test_ar)

combined_data = np.concatenate((train_ar,test_ar),axis=0)

test_inputs = []


for i in range(1238, 1258):
    test_inputs.append(combined_data[i:i+20, 0])


test_inputs = np.reshape(test_inputs,(len(test_inputs),xtrain.shape[1],1))
#predictions
predicted_vals = regressor.predict(test_inputs)

predicted_vals = scal.inverse_transform(predicted_vals)
test_ar = scal.inverse_transform(test_ar)


plt.plot(test_ar, color= 'green', label= 'actual stock price')
plt.plot(predicted_vals, color= 'blue', label= 'predicted stock price')
plt.title('Stock price analysis')
plt.xlabel('Time(days)')
plt.ylabel('Price')
plt.legend()
plt.show()



#Model 2 - 20 timesteps & 4 LSTM layers


xtrain=[] #array with t-20
ytrain=[] #array with t


for index in range(20,len(train_ar)):
    xtrain.append(train_ar[index-20:index,0])
    ytrain.append(train_ar[index,0])
    

xtrain= np.array(xtrain)
ytrain = np.array(ytrain)
    

xtrain = np.reshape(xtrain,(len(xtrain),xtrain.shape[1],1))

#Regressor
regressor= Sequential()
regressor.add(LSTM(units=4, return_sequences='True',input_shape= (None,1)))
regressor.add(LSTM(units=4, return_sequences='True'))
regressor.add(LSTM(units=4, return_sequences='True'))
regressor.add(LSTM(units=4))
regressor.add(Dense(units=1))
regressor.compile(optimizer ='rmsprop', loss= 'mean_squared_error')
regressor.fit(xtrain,ytrain,batch_size= 32, epochs=200)


test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_ar = test_data.iloc[:,1:2].values
test_ar = scal.transform(test_ar)

combined_data = np.concatenate((train_ar,test_ar),axis=0)

test_inputs = []


for i in range(1238, 1258):
    test_inputs.append(combined_data[i:i+20, 0])


test_inputs = np.reshape(test_inputs,(len(test_inputs),xtrain.shape[1],1))
#predictions
predicted_vals = regressor.predict(test_inputs)

predicted_vals = scal.inverse_transform(predicted_vals)
test_ar = scal.inverse_transform(test_ar)


plt.plot(test_ar, color= 'green', label= 'actual stock price')
plt.plot(predicted_vals, color= 'blue', label= 'predicted stock price')
plt.title('Stock price analysis')
plt.xlabel('Time(days)')
plt.ylabel('Price')
plt.legend()
plt.show()


#Evaluate RNN

rmse = math.sqrt(mean_squared_error(actual_price,predicted_vals))

print ("The root mean squared value of our model= {}".format(rmse) )

