import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
import torch.nn as nn
import torch
import torch.optim as optim

np.set_printoptions(suppress=True)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))


class AQINet(nn.Module):

    def __init__(self, n_features, time_steps, n_outputs):
        super(AQINet, self).__init__()
        self.n_features = n_features
        self.time_steps = time_steps
        self.n_outputs = n_outputs

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=6, batch_first=True)
        self.dense1 = nn.Linear(64 * 12, 128)
        self.dense2 = nn.Linear(128, 512)
        self.dense3 = nn.Linear(512, 1024)
        self.dense4 = nn.Linear(1024, 2048)
        self.dense5 = nn.Linear(2048, 4096)
        self.dense6 = nn.Linear(4096, 2048)
        self.dense7 = nn.Linear(2048, 512)
        self.dense8 = nn.Linear(512, 1024)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=n_outputs, num_layers=8)

    def forward(self, X):
        x, (h, c) = self.lstm1(X)
        x = torch.flatten(x, 1, 2)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = nn.Tanh()(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = nn.Tanh()(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = torch.reshape(x, (-1, 1, 1024))
        out, (h1, c1) = self.lstm2(x)
        return out


class AQINet2(nn.Module):

    def __init__(self, n_features, time_steps, n_outputs):
        super(AQINet2, self).__init__()
        self.n_features = n_features
        self.time_steps = time_steps
        self.n_outputs = n_outputs
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=256, num_layers=6, batch_first=True)
        self.fc1 = nn.Linear(256 * 12, 512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)
        self.active1 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 200)
        self.fc4 = nn.Linear(200, 1)
        self.active2 = nn.ELU()

    def forward(self, X):
        x, (h, c) = self.lstm1(X)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.active1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.active2(x)
        # print(x.shape)
        return x


def train(model, epochs, train_X, train_y, batch_size):
    criterion = My_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        step = 0
        while step * batch_size < train_X.shape[0]:
            end = train_X.shape[0]
            if (step + 1) * batch_size < train_X.shape[0]:
                end = (step + 1) * batch_size
            x = train_X[step * batch_size: end]
            y = train_y[step * batch_size: end]
            optimizer.zero_grad()
            out = model.forward(x)
            # print(out.shape, y.shape)
            # raise AttributeError("break")
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 2 == 0:
                print('epoch ' + str(epoch) + ' loss: %f' % loss.item())
    return model


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
    y = aqi_data_y[aqi_data_y.columns[target_columns.__len__() * 12:target_columns.__len__() * 12 + 1]]
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.9, shuffle=False)
    # # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    sample = train_X[0:1].values.reshape((-1, 12, feature_columns.__len__()))

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    test_X = scaler_X.transform(test_X)
    train_y = scaler_y.fit_transform(train_y)
    test_y = scaler_y.transform(test_y)
    train_X = torch.Tensor(train_X.reshape((-1, 12, 20))).cuda()
    train_y = torch.Tensor(train_y.reshape((-1, 1))).cuda()
    test_X = torch.Tensor(test_X.reshape((-1, 12, 20))).cuda()
    test_y = torch.Tensor(test_y.reshape(-1, 1)).cuda()
    # model = nn.Transformer(d_model=12, nhead=6, num_decoder_layers=2, num_encoder_layers=2).cuda()
    aqiNet = AQINet2(20, 12, 7).cuda()
    model = train(aqiNet, 50, train_X, train_y, 4096)
