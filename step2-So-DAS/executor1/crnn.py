#!/user/bin/env python3
  
import torch
import torch.nn as nn
import numpy as np

from sigprocess import STFT, ISTFT

class crnn(nn.Module):
    def __init__(self, fftsize=512, window_size=400, stride=100, channel=7):
        super(crnn, self).__init__()
        bins = fftsize // 2
        self.channel = channel
        self.dropout_fc = 0

        self.stft = STFT(fftsize=fftsize, window_size=window_size, stride=stride, trainable=False)

        self.input_conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(2*channel,64,[3,3],[1,1],padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d([8,1]),
            torch.nn.Dropout(p=self.dropout_fc),
        )
        self.input_conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,[3,3],[1,1],padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d([4,1]),
            torch.nn.Dropout(p=self.dropout_fc),
        )
        self.input_conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,[3,3],[1,1],padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d([4,1]),
            torch.nn.Dropout(p=self.dropout_fc),
        )
        self.blstm = torch.nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_fc,
            bidirectional=True
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128,429),
            torch.nn.Linear(429,2),
        )

    def forward(self, x):    # b m n
        xs = self.stft(x[:,[0],:])[...,1:,:].unsqueeze(1) # b 1 t f 2
        B, T = xs.size(0), xs.size(2)
        for i in range(1,self.channel):
            xs = torch.cat((xs,self.stft(x[:,[i],:])[...,1:,:].unsqueeze(1)),1) # b c t f 2
        feat_in = torch.cat((xs[...,0], xs[...,1]), 1).permute(0,1,3,2)         # b 2c f t

        h = self.input_conv_layer1(feat_in)
        h = self.input_conv_layer2(h)
        h = self.input_conv_layer3(h)
        h = h.permute(0,2,1,3).contiguous().view(B,T,128)
        h, _ = self.blstm(h)
        h = self.fc_layer(h) # b t 4

        est_xy = h.mean(1) # b 2

        return est_xy
