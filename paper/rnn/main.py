import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonLinearity='tanh', bias=True, batch_first=False, dropout=0., bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size # input dimension
        self.hidden_size = hidden_size # hidden dimension
        self.num_layers = num_layers # number of layers
        self.nonLinearity = nonLinearity # nonlinearity
        self.bias = bias # bias
        self.batch_first = batch_first # batch first
        self.dropout = dropout # dropout
        self.bidirectional = bidirectional # bidirectional
        self.num_directions = 2 if self.bidirectional else 1 # number of directions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device

    def forward(self, input, hx=None):
        h_0 = torch.zeros(self.num_layers * self.num_directions, input.size(0), self.hidden_size).to(self.device) # initial hidden state
        c_0 = torch.zeros(self.num_layers * self.num_directions, input.size(0), self.hidden_size).to(self.device) # initial cell state

        # input, hidden, and cell states
        input = input.to(self.device)
        hx = hx.to(self.device) if hx is not None else None
        c0 = c0.to(self.device) if c0 is not None else None
