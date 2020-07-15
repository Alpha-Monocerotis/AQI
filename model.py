import torch
import torch.nn as nn


class AQINet(nn.Module):

    def __init__(self, n_features,  time_steps, n_outputs):
        super(AQINet, self).__init__()
        self.n_features = n_features
        self.time_steps = time_steps
        self.n_outputs = n_outputs

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=time_steps, num_layers=6, batch_first=True)
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 512)
        self.dense3 = nn.Linear(512, 128)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=n_outputs)
        # self.encoder_l1 = nn.TransformerEncoderLayer(d_model=n_features, nhead=8, dim_feedforward=512, activation='tanh')
        # self.encoder_l2 = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, activation='tanh')
        # self.encoder_l3 = nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, activation='tanh')
        #
        # self.decoder_l1 = nn.TransformerDecoderLayer(d_model=2048, nhead=8, dim_feedforward=1024, activation='tanh')
        # self.decoder_l2 = nn.TransformerDecoderLayer(d_model=1024, nhead=8, dim_feedforward=512, activation='tanh')
        # self.decoder_l3 = nn.TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=n_outputs, activation='softmax')


    def forward(self, X):
        X = self.lstm1(X)
        print(X)

if __name__ == '__main__':
    pass

