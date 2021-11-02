"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch.nn as nn
from dgl.nn.pytorch import EdgeGraphConv
from dgl import readout_nodes


class EGNNC(nn.Module):  # Create GCN class
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0.05
                 ):
        """

        :param in_feats: number of input features
        :param n_hidden: number of features in the hidden layers
        :param n_classes: number of features in the output layer
        :param n_layers: TOTAL number of layers in the network
        :param activation: activation function to be used
        :param dropout:
        """
        super(EGNNC, self).__init__()

        self.layers = nn.ModuleList()  # Create empty list of layers
        # input layer
        self.layers.append(EdgeGraphConv(in_feats, n_hidden, norm="left", activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(EdgeGraphConv(n_hidden, n_hidden, norm="left", activation=activation))
        # output layer
        self.layers.append(EdgeGraphConv(n_hidden, n_classes, norm="left"))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, w):
        '''
        features pass to j
        b_centralities, d_centralities, torch.Tensor(self.edge_vector).unsqueeze(-1)), dim=1
        '''
        h = g.ndata['features']  # Get features from graph
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=w)  # Features after they been convoluted
        g.ndata['h'] = h
        mN = readout_nodes(g, 'h', op="mean")  # mean of those features
        g.ndata.pop('h')
        return h, mN
