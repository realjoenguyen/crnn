from typing import List

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models
import numpy as np
from torch.nn import init
import config

class CRNN(nn.Module):
    def __init__(self,
                 input_size,
                 abc,
                 backend,
                 lstm_hidden_size=config.lstm_hidden_size,
                 lstm_num_layers=config.lstm_num_layers,
                 lstm_dropout=config.dropout,
                 seq_proj=[0, 0],
                 ):
        super(CRNN, self).__init__()

        self.abc = abc
        self.num_classes = len(self.abc) + 1 # include blank id = 0
        self.input_size = input_size
        feature_extractor = getattr(models, backend)(pretrained=True)
        self.resnet18 = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            # feature_extractor.layer4
        )
        self.downrate = config.downrate
        self.num_filter = config.num_filter
        self.feature_dim = int(input_size[1] / self.downrate) * self.num_filter
        self.lstm_input_size= config.lstm_input_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        # self.fully_conv = seq_proj[0] == 0
        # if not self.fully_conv:
        #     self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)
        # self.rnn = nn.LSTM(self.get_block_size(self.cnn),

        self.lstm = nn.LSTM(self.lstm_input_size,
                            lstm_hidden_size, self.lstm_num_layers,
                            batch_first=False,
                            dropout=lstm_dropout, bidirectional=True)

       # transformation
        self.cnn2lstm = nn.Sequential(
            nn.Linear(self.feature_dim, self.lstm_input_size),
            nn.ReLU(),
        )
        self.lstm2logit = nn.Linear(lstm_hidden_size * 2, self.num_classes)
        self.softmax = nn.Softmax(dim=2)

        # (len, batch, dim)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        # transformation
        init.xavier_uniform_(self.cnn2lstm[0].weight, gain=gain)
        init.constant_(self.cnn2lstm[0].bias, 0.)
        init.xavier_uniform_(self.lstm2logit.weight, gain=gain)
        init.constant_(self.lstm2logit.bias, 0.)

        # lstm
        for pname, pval in self.lstm.named_parameters():
            if pname.startswith('weight'):
                init.orthogonal_(pval)
            else:
                assert pname.startswith('bias')
                init.constant_(pval, 0.)

    def forward(self, x, decode=False, debug=False):
        # x = (b, c, input_h, input_w)
        features = self.resnet18(x) # (b, c, h, w)
        # TODO: add attention on top of CNN

        # (w, b, feature_map)
        features = self.features_to_sequence(features)

        # seq = (w, b, 2 * dim)
        seq, hidden = self.lstm(features)

        # seq = (w, b, num_class)
        seq = self.lstm2logit(seq)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                seq = self.decode(seq)
        else:
            softmax = self.softmax(seq)
            seq = self.log_softmax(seq)
            if debug:
                seq = seq, softmax
        return seq

    # def init_hidden(self, batch_size, gpu=False):
    #     h0 = Variable(torch.zeros(self.lstm_num_layers * 2,
    #                               batch_size,
    #                               self.lstm_hidden_size))
    #     h0 = h0.cuda()
    #     return h0

    def features_to_sequence(self, features):
        # b, c, h, w = features.size()
        # assert h == 1, "the height of out must be 1"
        # if not self.fully_conv:
        #     features = features.permute(0, 3, 2, 1) # (b, w, h, c)
        #     features = self.proj(features)
        #     features = features.permute(1, 0, 2, 3) # (w, b, h, c)
        # else:
        #     features = features.permute(3, 0, 2, 1) # (w, b, h, c)
        # features = features.squeeze(2)

        features = features.permute(3, 0, 2, 1) # (w, b, h, c)
        features = features.contiguous().view(features.size()[0], features.size()[1], -1)  # (w, b, h * c)
        features = self.cnn2lstm(features) # (w, b, rnn_input)
        return features

    # def get_block_size(self, layer):
    #     return layer[-1][-1].bn2.weight.size()[0]

    def pred_to_string(self, pred):
        # pred.shape = (w, dim)
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        out = ''.join(self.abc[i] for i in out)
        return out

    def decode(self, pred):
        # TODO: add this one https://github.com/githubharald/CTCDecoder
        # (w, b, dim) -> (b, w, dim)
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq
