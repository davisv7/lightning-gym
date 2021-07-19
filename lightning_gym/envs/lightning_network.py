from gym import Env, logger, spaces
from gym.spaces import Discrete, Box
import numpy as np
import networkit as nk
from bidict import bidict
from ..utils import *
from ..GCN import GCN
import dgl
import torch
import torch.nn.functional as F
import random

'''
    Get the current directory.
    Join the current directory with the name of file. Sample_Snapshots
'''

CWD = getcwd()
SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')


# Environment Class
class NetworkEnvironment(Env):

    def __init__(self, budget=10, node_id=None):
        """
        :param budget:
        """
        self.budget = budget  # number of channels that can add to graph
        self.action_space = None
        self.observation_space = Box(low=np.array([0]), high=np.array([2]))
        self.node_id = node_id
        self.node_index = None
        self.index_to_node = bidict()
        self.nk_g = None  # Networkit graph
        self.dgl_g = None  # Make DGL graph
        self.graph_size = None
        self.dyn_btwn_getter = None
        self.btwn_cent = 0  # Betweens
        self.gcn = None
        self.edge_vector = None
        self.features = None
        self.nodes = None
        self.edges = None

    def get_features(self):

        '''     Initialize Algorithm
        Measures the extent to which a node lies on shortest
        paths between other nodes. Nodes with high betweeness
        could have considerable more influce.
        '''

        bc = nk.centrality.Betweenness(self.nk_g, )  # Pa
        bc.run()  # Run the algorithm
        b_centralities = torch.Tensor(bc.scores()).unsqueeze(-1)
        # print(b_centralities)
        '''Initialize Algorithm
        Indicates how close a node is to all other nodes in the network. 
        '''
        dc = nk.centrality.DegreeCentrality(self.nk_g)
        # Run the algorithm
        dc.run()
        d_centralities = torch.Tensor(dc.scores()).unsqueeze(-1)
        # print(d_centralities)

        # ???
        self.features = torch.cat((b_centralities, d_centralities, torch.Tensor(self.edge_vector).unsqueeze(-1)), dim=1)
        # x = 1

    def update_features(self):
        col_idx = torch.Tensor(np.repeat(-1, self.features.size(0))).long()
        rows = torch.arange(self.features.size(0)).long()
        update_values = torch.FloatTensor(self.edge_vector)

        self.features[rows, col_idx] = update_values
        self.dgl_g.ndata["features"] = self.features

    def step(self, action: int):
        done = False
        # calculate reward
        if self.edge_vector[action] == 1:
            reward = 0
            done = True
        else:
            # what if, by adding a node twice, we remove it? increase budget, reward is negative change
            # we would have to reinit the betweenness calculator on edge removals, which isn't terrible
            self.edge_vector[action] = 1
            reward = self.get_reward(action)

        # check if done with budget
        if sum(self.edge_vector) == self.budget:
            done = True

        info = {}
        return self.gcn(self.dgl_g), reward, done, info

    def get_reward(self, action):
        neighbor_index = action
        event_type = 3  # nk.dynamic.GraphEvent.EDGE_ADDITION = 3

        # add an edge in one direction
        self.nk_g.addEdge(self.node_index, neighbor_index, w=1)
        event = nk.dynamic.GraphEvent(event_type, self.node_index, neighbor_index, 1)
        self.dyn_btwn_getter.update(event)

        # and another in the other direction
        self.nk_g.addEdge(neighbor_index, self.node_index, w=1)
        event = nk.dynamic.GraphEvent(event_type, neighbor_index, self.node_index, 1)
        self.dyn_btwn_getter.update(event)
        new_btwn = self.dyn_btwn_getter.getbcx() / (self.graph_size * (self.graph_size - 1) / 2)
        reward = new_btwn - self.btwn_cent
        self.btwn_cent = new_btwn
        return reward

    def reset(self):
        # gets random file name, loads graphs, returns state
        randomfilename = get_random_filename()
        nodes, edges = load_json(path.join(SAMPLEDIRECTORY, randomfilename))
        self.nodes = nodes  # Added nodes
        self.edges = edges  # Added edges
        # Create nx_graph
        nx_graph = make_nx_graph(nodes, edges)
        self.dgl_g = dgl.from_networkx(nx_graph)
        self.graph_size = len(nx_graph.nodes())

        self.get_edge_vector_from_node()

        # Create tuples index : pubKey into bidictionary
        self.index_to_node = bidict(enumerate(nx_graph.nodes()))
        self.index_to_node = bidict(enumerate(nx_graph.nodes()))
        # Assing network kit graph
        self.nk_g = nx_to_nk(nx_graph, self.index_to_node)

        self.get_features()

        self.dgl_g.ndata['features'] = self.features

        self.gcn = GCN(
            in_feats=3,
            n_hidden=4,
            n_classes=3,
            n_layers=2,
            activation=F.relu)

        if self.node_id is None:
            # self.index_to_node[self.graph_size] = self.node_id
            self.node_index = self.graph_size
            self.nk_g.addNode()

        self.dyn_btwn_getter = nk.centrality.DynBetweennessOneNode(self.nk_g, self.node_index)
        self.dyn_btwn_getter.run()
        self.btwn_cent = self.dyn_btwn_getter.getbcx() / ((self.graph_size - 1) * (self.graph_size - 2) / 2)

        done = False
        self.action_space = Discrete(self.graph_size)

        return self.gcn(self.dgl_g), done

    # render on frame of environment at a time

    # Given public ID number from a node
    def get_edge_vector_from_node(self):
        # Create a vector of zeros to the length of the graph_size
        self.edge_vector = [0 for _ in range(self.graph_size)]

        if self.node_id is not None:
            for edge in self.edges:
                if edge[0] == self.node_id:
                    neighbor_index = self.index_to_node.inverse[edge[1]]
                    self.edge_vector[neighbor_index] = 1

        return self.edge_vector

    def get_random_node_key(self):
        self.nodes
        random_key = random.choice(self.nodes)
        return random_key

    def render(self, mode='channel'):
        pass
