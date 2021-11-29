"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl import readout_nodes


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 n_layers,
                 activation,
                 dropout=0.10
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

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hid_feats, norm="left", activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(hid_feats, hid_feats, norm="left", activation=activation))
        # output layer
        self.layers.append(GraphConv(hid_feats, out_feats, norm="left"))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, w=None):
        """
        Forward function defines how data is passed through the neural network.
        :param g: graph itself (dgl graph)
        :return: h tensor of node out-features, mN the column-wise mean of these features
        """
        h = g.ndata['features']  # Get features from graph
        for i, layer in enumerate(self.layers):
            # if i != len(self.layers) - 1:
            #     h = self.dropout(h)
            h = layer(g, h)  # Features after they been convoluted, these represent the nodes
        g.ndata['h'] = h
        mN = readout_nodes(g, 'h', op="mean")  # column-wise average of those node features, this represents the graph
        return h, mN
