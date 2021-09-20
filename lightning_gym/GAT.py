# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn.pytorch.conv import GATConv
from dgl import readout_nodes
#
#
# class GAT(torch.nn.Module):
#     def __init__(self,
#                  num_features=1,
#                  hid=8,
#                  in_head=1,
#                  out_head=1,
#                  num_classes=2
#                  ):
#         super(GAT, self).__init__()
#         self.num_features = num_features
#         self.hid = hid
#         self.in_head = in_head
#         self.out_head = out_head
#         self.num_classes = num_classes
#
#         self.policy = nn.Linear(num_classes, 1,)  # We know is the output of the GCN
#         self.value = nn.Linear(num_classes, 1, )  # Output?
#
#         self.conv1 = GATConv(self.num_features, self.hid, num_heads=self.in_head, feat_drop=0.1)
#         self.conv2 = GATConv(self.hid * self.in_head, self.num_classes, num_heads=self.out_head, feat_drop=0.1)
#
#     def forward(self, g):
#         g,h = g,g.ndata["features"]
#         # g.edata["e"] = g.edata["weight"]
#
#         # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
#         # One can skip them if the dataset is sufficiently large.
#
#         h = F.dropout(h, p=0.1, training=self.training)
#         h = self.conv1(g, h).flatten(1)
#         h = F.leaky_relu(h)
#         h = F.dropout(h, p=0.1, training=self.training)
#         h = self.conv2(g, h).squeeze()
#
#         g.ndata['h'] = h
#         mN = readout_nodes(g, 'h', op="mean")  # Sum of those three features
#         PI = self.policy(h)  # Distribution of actions
#         V = self.value(mN)
#         g.ndata.pop('h')
#         return PI, V  # Use it in the run episode method
#

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()

        self.policy = nn.Linear(num_classes, 1,)  # We know is the output of the GCN
        self.value = nn.Linear(num_classes, 1, )  # Output?

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g):
        h=g.ndata["features"]
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](g, h).mean(1)
        # return logits
        g.ndata['h'] = h
        mN = readout_nodes(g, 'h', op="mean")  # Sum of those three features
        PI = self.policy(h)  # Distribution of actions
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V  # Use it in the run episode method


# from scipy.sparse import lil_matrix
#
#
# def preprocess_attention(edge_atten, g, to_normalize=True):
#     """Organize attentions in the form of csr sparse adjacency
#     matrices from attention on edges.
#     Parameters
#     ----------
#     edge_atten : numpy.array of shape (# edges, # heads, 1)
#         Un-normalized attention on edges.
#     g : dgl.DGLGraph.
#     to_normalize : bool
#         Whether to normalize attention values over incoming
#         edges for each node.
#     """
#     n_nodes = g.number_of_nodes()
#     num_heads = edge_atten.shape[1]
#     all_head_A = [lil_matrix((n_nodes, n_nodes)) for _ in range(num_heads)]
#     for i in range(n_nodes):
#         predecessors = list(g.predecessors(i))
#         edges_id = g.edge_ids(predecessors, i)
#         for j in range(num_heads):
#             all_head_A[j][i, predecessors] = (
#                 edge_atten[edges_id, j, 0].data.cpu().numpy()
#             )
#     if to_normalize:
#         for j in range(num_heads):
#             all_head_A[j] = normalize(all_head_A[j], norm="l1").tocsr()
#     return all_head_A


# # Take the attention from one layer as an example
# # num_edges x num_heads x 1
# A = self.g.edata["a_drop"]
# # list of length num_heads, each entry is csr of shape (num_nodes, num_nodes)
# A = preprocess_attention(A, self.g)

# if __name__ == "__main__":
#     pass