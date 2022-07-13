
import math
import torch.nn as nn

_XAVIER_INIT_ = False

def LSTMEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if not _XAVIER_INIT_:
        nn.init.uniform_(m.weight, -0.1, 0.1)
    else:
        nn.init.xavier_normal_(m.weight)

    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)  # NOTE: use this default initialization for SLU and as default
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            if not _XAVIER_INIT_:
                param.data.uniform_(-0.1, 0.1)
            else:
                if len(param.data.size()) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.data.uniform_(-math.sqrt(3/hidden_size), math.sqrt(3/hidden_size))
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)  # NOTE: use this default initialization for SLU and as default
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            if not _XAVIER_INIT_:
                param.data.uniform_(-0.1, 0.1)
            else:
                if len(param.data.size()) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.data.uniform_(-math.sqrt(3/hidden_size), math.sqrt(3/hidden_size))
    return m

def SLULSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)  # NOTE: use this default initialization for SLU and as default
    '''for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            if not _XAVIER_INIT_:
                param.data.uniform_(-0.1, 0.1)
            else:
                if len(param.data.size()) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.data.uniform_(-math.sqrt(3/hidden_size), math.sqrt(3/hidden_size))'''
    return m


def SLULSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)  # NOTE: use this default initialization for SLU and as default
    '''for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            if not _XAVIER_INIT_:
                param.data.uniform_(-0.1, 0.1)
            else:
                if len(param.data.size()) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.data.uniform_(-math.sqrt(3/hidden_size), math.sqrt(3/hidden_size))'''
    return m

def LSTMLinear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    if not _XAVIER_INIT_:
        m.weight.data.uniform_(-0.1, 0.1)
    else:
        nn.init.xavier_uniform_(m.weight)
    if bias:
        if not _XAVIER_INIT_:
            m.bias.data.uniform_(-0.1, 0.1) 
        else:
            m.bias.data.uniform_(-math.sqrt(3/out_features), math.sqrt(3/out_features)) 
    return m

def LSTMConv1d(in_features, out_features, kernel_size, stride=1):

    c = nn.Conv1d(in_features, out_features, kernel_size, stride=stride)
    if not _XAVIER_INIT_:
        c.weight.data.uniform_(-0.1, 0.1)
        c.bias.data.uniform_(-0.1, 0.1)
    else:
        nn.init.xavier_uniform_(c.weight) 
        c.bias.data.uniform_(-math.sqrt(3/out_features), math.sqrt(3/out_features))
    return c

def TransformerEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def TransformerLinear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def TransformerConv1d(in_features, out_features, kernel_size, stride=1):

    c = nn.Conv1d(in_features, out_features, kernel_size, stride=stride)
    nn.init.xavier_uniform_(c.weight)
    nn.init.constant_(c.bias, 0.0)
    return c

def TransformerLSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            #param.data.uniform_(-0.1, 0.1)
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    return m


def TransformerLSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            #param.data.uniform_(-0.1, 0.1)
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
    return m

