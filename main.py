import numpy as np
import pandas as pd
import matplotlib as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

#Fiancial prodcut being tracked
financial_product = 'BTC'
#Currency the finacial product is to be tracked to
against_currency = 'CAD'

#date range for data
start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{financial_product}-{against_currency}', 'yahoo', start, end)

#prepare data for machine learning
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#number of prediction days
prediction_days = 60

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#create neaural network

