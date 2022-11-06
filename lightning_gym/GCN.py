"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from dgl import mean_nodes
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 n_layers,
                 activation,
                 ):
        """
        Graph Convolutional Network Class
        :param in_feats: number of input features
        :param n_hidden: number of features in the hidden layers
        :param n_classes: number of features in the output layer
        :param n_layers: TOTAL number of layers in the network
        :param activation: activation function to be used
        :param dropout:
        """
        super(GCN, self).__init__()
        self.policy = nn.Linear(out_feats, 1)
        self.value = nn.Linear(out_feats, 1)

        self.layers = nn.ModuleList()
        if n_layers == 1:
            hid_feats = out_feats
            activation = None

        # hidden layers
        for i in range(n_layers):
            if i == 0:
                # input layer
                self.layers.append(GraphConv(in_feats, hid_feats, norm="left", activation=activation))
            elif i != n_layers - 1:
                # hidden layer
                self.layers.append(GraphConv(hid_feats, hid_feats, norm="left", activation=activation))
            else:
                # output layer
                self.layers.append(GraphConv(hid_feats, out_feats, norm="left"))

    def forward(self, g, w=None):
        """
        Forward function defines how data is passed through the neural network.
        :param w: tensor of edge weights
        :param g: graph itself (dgl graph)
        :return: h tensor of node out-features, mN the column-wise mean of these features
        """
        h = deepcopy(g.ndata['features'])  # Get features from graph
        norm = EdgeWeightNorm(norm='left')
        scaler = MinMaxScaler((0, 1))
        for i, layer in enumerate(self.layers):
            if w is not None:
                # w = norm(g, w.squeeze())
                w = 1 / (w + 1)
                w = torch.Tensor(scaler.fit_transform(w.reshape(-1, 1)))
                h = layer(g, h, edge_weight=w)
            else:
                h = layer(g, h)  # Features after they been convoluted, these represent the nodes
        g.ndata['h'] = h
        mN = mean_nodes(g, 'h')  # column-wise average of those node features, this represents the graph
        PI = self.policy(h)
        V = self.value(mN)
        return PI, V
