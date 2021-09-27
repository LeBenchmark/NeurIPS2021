import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class myLinTanh(nn.Module):
    def __init__(self, featSize, lastOnly=False, device="cuda", outputSize=1):
        super(myLinTanh, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.linTh = nn.Sequential(
            nn.Linear(featSize, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.linTh(x)
        return output
        
class myGRU(nn.Module):
    def __init__(self, featSize, hidden_size=128, num_layers=1, device="cuda", lastOnly=False, outputSize=1):
        super(myGRU, self).__init__()

        self.featSize = featSize
        self.lastOnly = lastOnly
        self.outputSize = outputSize

        self.rnn = nn.GRU(
            input_size=featSize, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.linTh = nn.Sequential(
            nn.Linear(hidden_size, outputSize),
            nn.Tanh(),
        )

    def forward(self, x):
        # batch_size, timesteps, sq_len = x.size()
        # x = x.view(batch_size, timesteps, sq_len)
        output, _ = self.rnn(x)
        if self.lastOnly: output = output[:, -1, :].unsqueeze(1)
        output = self.linTh(output)
        return output

