import numpy as np
import pandas as pd
import matplotlib as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

financial_product = 'BTC'
against_currency = 'CAD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{financial_product}-{against_currency}', 'yahoo', start, end)

#prepare data for machine learning
print(data.head())