"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl import readout_nodes


class GCN(nn.Module):
    def __init__(self,
                 in_feats,  # Number of features each node has
                 n_hidden,  # Size of Hidden Features (Neighbors features)
                 n_classes,  # Size of final features vector
                 n_layers,  # Number fo layers
                 activation  # Activation layer Relu in our case
                 # dropout
                 ):
        super(GCN, self).__init__()

        #Why Policy and value
        self.policy = nn.Linear(n_classes, 1) #We know is the output of the GCNN
        self.value = nn.Linear(n_classes, 1, )
        self.layers = nn.ModuleList()  # Create empty list of layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)

    # self.gcn.forward(self.dgl_g), torch.Tensor([reward])
    def forward(self, g):
        '''
        features pass to j
        b_centralities, d_centralities, torch.Tensor(self.edge_vector).unsqueeze(-1)), dim=1
        '''
        h = g.ndata['features'] #Get features
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(g, h) #Features after they been convoluted
        g.ndata['h'] = h
        mN = readout_nodes(g, 'h') #Sum of those three features
        PI = self.policy(h)       #Distribution of actions
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V #Use it in the run episod method


