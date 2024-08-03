'''
numpy
Array interface is the best and the most important feature of Numpy.

Pandas:
loading and manipulating the datasets

keras:
Keras also provides some of the best utilities for compiling models, 
processing data-sets, visualization of graphs, and much more.

TensorFlow works like a computational library for writing new algorithms that involve 
a large number of tensor operations,
since neural networks can be easily expressed as computational graphs

sklearnIt is a Python library is associated with NumPy
It is considered as one of the best libraries for working with complex data.

Command to run the File:
py -m streamlit run main.py
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import streamlit as st


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import  Sequential

st.title('Stock and CryptOcurrency Price Prediction')

stock_name = st.selectbox('Trending Stocks',('MRF.NS','TCS.NS', 'AMD', 'GVN', 'BB', 'INFY.NS', 'LTI.NS','APPL','BTC-USD','ETH-USD'))
#st.write('You selected:', Host_Country)
#stock_name=st.text_input("Enter Any Stock",'TCS.NS')
future_day=st.number_input("Enter Number of Days",1)



#against_currency='INR'

start = dt.datetime(2018,1,1)
end = dt.datetime.now()
print(end)
# Taking data from web using yahoo finanace API
data = web.DataReader(f'{stock_name}','yahoo',start,end)

# data = pd.read_excel('LIVE DATA.xlsx', engine='openpyxl')
# #prepare data
# print(data)
# start_date = "2020-01-1"
# end_date = "2022-04-30"
# data1 = web.DataReader(name="AMD", data_source='yahoo', start=start_date, end=end_date)
# #feature scaling
# scalar = MinMaxScaler(feature_range=(0,1))
# scaled_data=scalar.fit_transform(data['Close'].values.reshape(-1,1))

# prediction_days=14
# #future_day=2


# x_train,y_train=[],[]
# for x in range(prediction_days,len(scaled_data)-future_day):
#     x_train.append(scaled_data[x-prediction_days:x,0])
#     y_train.append(scaled_data[x+future_day,0])

# x_train,y_train= np.array(x_train),np.array(y_train)
# x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# #CREATING A NEURAL NETWORK

# model=Sequential()
# model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.4))
# model.add(LSTM(units=50,return_sequences=True))
# model.add(Dropout(0.4))
# model.add(LSTM(units=50))
# model.add(Dropout(0.5))
# model.add(Dense(units=1))

# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(x_train,y_train,epochs=10,batch_size=32)

# #TESTING THE MODEL

# test_start=dt.datetime(2021,1,1)
# test_end=dt.datetime.now()
# #print(test_end)
# test_data = web.DataReader(f'{stock_name}','yahoo',test_start,test_end)
# actual_prices=test_data['Close'].values

# total_dataset=pd.concat((data['Close'],test_data['Close']),axis=0)

# model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
# model_inputs=model_inputs.reshape(-1,1)
# model_inputs=scalar.fit_transform(model_inputs)

# x_test=[]

# for x in range(prediction_days,len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_days:x,0])

# x_test=np.array(x_test)
# x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# prediction_prices=model.predict(x_test)
# prediction_prices=scalar.inverse_transform(prediction_prices)


# #plotting the graph

# plt.plot(actual_prices,color='red',label='Predicted Price')
# plt.plot(prediction_prices,color='black',label='Actual Price')


# plt.title('Stock Market Price Prediction')
# plt.title(stock_name +" Stock Market Price Prediction")
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend(loc='upper left')
# plt.show()

# #PREDICT THE NEXT DAY

# real_data=[model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs)+1,0]]     
# real_data=np.array(real_data)
# real_data=np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

# prediction=model.predict(real_data)
# prediction= scalar.inverse_transform(prediction)
# print(stock_name +" Predicted Price is : Rs.",prediction)


# #describing data


# #st.write(data.describe())

# st.subheader(stock_name + ' Predicted Price of is Rs.' )
# st.subheader(prediction)
# #st.subheader(data1)
# fig =plt.figure(figsize=(12,6))
# plt.plot(data.Close)
# st.pyplot(fig)


# #plt.plot(actual_prices,color='red',label='Predicted Price')
# #plt.plot(prediction_prices,color='black',label='Actual Price')


# st.subheader('Predicted Price vs Original Price')
# ma100 = data.Close.rolling(100).mean()

# fig =plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(data.Close)
# st.pyplot(fig)

# st.subheader('Summary')
# st.write(data.describe())

# st.subheader('Dataset')
# st.write(data)
