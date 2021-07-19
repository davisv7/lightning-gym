"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats, # Number of features each node has
                 n_hidden, # Size of Hidden Features (Neighbors features)
                 n_classes, #Size of final features vector
                 n_layers, # Number fo layers
                 activation # Actication layer Relu in our case
                 #dropout
                 ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList() #Create empty list of layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        h = g.ndata['features']
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(g, h)
        return h