import pandas as pd
import numpy as np
from model import AQINet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
import torch.nn as nn
import torch


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == '__main__':
    PATH = './all_data.csv'
    aqi_data = pd.read_csv(PATH)
    aqi_data = aqi_data.drop(['Unnamed: 0', 'date'], axis=1)
    feature_columns = aqi_data.columns
    target_columns = ['AQI', 'PM2_5', 'PM_10', 'SO2', 'NO2', 'O3', 'CO']
    aqi_data_supervised = series_to_supervised(aqi_data, 12, 12, True)
    X = aqi_data_supervised[aqi_data_supervised.columns[0: feature_columns.__len__() * 12]]
    aqi_data_y = series_to_supervised(aqi_data[target_columns], 12, 12, True)
    y = aqi_data_y[aqi_data_y.columns[target_columns.__len__() * 12:]]
    X.index = list(range(X.shape[0]))
    y.index = list(range(y.shape[0]))
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    sample = train_X[0:1].values.reshape((-1, 12, feature_columns.__len__()))

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    test_X = scaler_X.transform(test_X)
    train_y = scaler_y.fit_transform(train_y)
    train_X = train_X.reshape((-1, 12, 20))
    train_y = train_y.reshape((-1, 12, 7))
    print(train_X.shape, train_y.shape)
    # model = nn.Transformer(d_model=12, nhead=6, num_decoder_layers=2, num_encoder_layers=2).cuda()
    aqiNet = AQINet(20, 12, 7)
    aqiNet.forward(train_X)
    # train_X = torch.Tensor(train_X).cuda()
    # train_y = torch.Tensor(train_y).cuda()
    # model(train_X, train_y)


