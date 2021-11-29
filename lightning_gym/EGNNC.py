import torch.nn as nn
from dgl.nn.pytorch import EdgeGraphConv
from dgl import readout_nodes


class EGNNC(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0.10
                 ):
        """
        Edge Exploiting Graph Neural Network
        :param in_feats: number of input features
        :param n_hidden: number of features in the hidden layers
        :param n_classes: number of features in the output layer
        :param n_layers: TOTAL number of layers in the network
        :param activation: activation function to be used
        :param dropout:
        """
        super(EGNNC, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(EdgeGraphConv(in_feats, n_hidden, norm="left", activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(EdgeGraphConv(n_hidden, n_hidden, norm="left", activation=activation))
        # output layer
        self.layers.append(EdgeGraphConv(n_hidden, n_classes, norm="left"))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, w):
        """
        Forward function defines how data is passed through the neural network.
        :param g: graph itself (dgl graph)
        :param w: tensor of edge weights
        :return: h tensor of node out-features, mN the column-wise mean of these features
        """
        h = g.ndata['features']  # Get features from graph
        for i, layer in enumerate(self.layers):
            # if i != len(self.layers)-1:
            #     h = self.dropout(h)
            h = layer(g, h, edge_weight=w)  # Features after they been convoluted, these represent the nodes
        g.ndata['h'] = h
        mN = readout_nodes(g, 'h', op="sum")  # column-wise average of those node features, this represents the graph
        g.ndata.pop('h')
        return h, mN
