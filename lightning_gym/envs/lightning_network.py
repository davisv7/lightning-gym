import networkx as nx
from gym import Env
import numpy as np
from bidict import bidict
from ..utils import *
import dgl
import torch
import random
from ..Logger import Logger
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
from ..graph_utils import *
from functools import reduce

'''
    Get the current directory.
    Join the current directory with the name of file. Sample_Snapshots
'''

CWD = getcwd()
SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')


# Environment Class
class NetworkEnvironment(Env):

    def __init__(self, budget=10, node_id=None, **kwargs):
        """
        :param
        budget: number of channels that can add to graph
        node_index: node parameter that want to test on
        ig_g: igraph graph
        dgl_g: dgl graph
        btwn_cent: betweeness centrality
        edge_vector: neighbors of a node, (1=neighbor, 0=not neighbor)
        features: betweeness, degree, closeness, edge vector
        r_logger: keeps track of reward
        repeat: whether or not to repeat
        budget_offset: how many edges the nodes started off with
        nx_graph: networkX graph
        k: size of subgraph
        params = kwards: pass keywords
        """
        self.budget = budget
        self.node_id = node_id
        self.index_to_node = bidict()
        self.r_logger = Logger()
        self.node_index = None
        self.ig_g = None
        self.dgl_g = None
        self.graph_size = None
        self.btwn_cent = 0
        self.edge_vector = None
        self.features = None
        self.budget_offset = 0
        self.nx_graph = None
        self.base_graph = None
        self.norm = None
        self.num_actions = 0
        self.k = kwargs.get("k", None)
        self.repeat = kwargs.get("repeat", False)
        self.graph_type = kwargs.get('graph_type', 'scale_free')
        assert self.graph_type in ['sub_graph', 'snapshot', 'scale_free'], \
            """You must use one of the following graphs types:
                                                        sub_graph, 
                                                        snapshot,
                                                        scale_free"""

    def __str__(self):
        return \
            """Configurations:
              "Graph Type: {}
              "Budget: {}
              "Repeat State: {}""".format(self.graph_type, self.budget, self.repeat)

    def get_features(self):
        '''
        Initialize Algorithm
        Measures the extent to which a node lies on shortest
        paths between other nodes. Nodes with high betweeness
        could have considerable more influence.
        '''
        weights = self.ig_g.es["weight"]
        norm = (self.graph_size * (self.graph_size - 1) / 2)
        w_norm = sum(weights)
        indices = self.ig_g.vs().indices
        b_centralities = np.array(self.ig_g.betweenness(indices, weights=weights)) / norm
        b_centralities = torch.Tensor(b_centralities).unsqueeze(-1)  # makes list smaller size

        '''Initialize Algorithm
        Indicates how close a node is to all other nodes in the network. 
        '''
        dc = np.zeros(shape=self.graph_size)
        for i, node in enumerate(self.ig_g.vs()):
            incident_weights = self.ig_g.es.select(_source=[node])["weight"]
            dc[i] = sum(incident_weights) / w_norm

        d_centralities = torch.Tensor(dc).unsqueeze(-1)

        cc = self.ig_g.closeness(range(self.graph_size))
        c_centralities = torch.Tensor(cc).unsqueeze(-1).nan_to_num(0)

        lc = nx.load_centrality(undirected(self.nx_graph))
        l_centralities = torch.Tensor([lc[node["name"]] for node in self.ig_g.vs()]).unsqueeze(-1)

        'appending 3 features in a tensor'
        self.features = torch.cat((
            b_centralities,
            d_centralities,
            c_centralities,
            l_centralities,
            self.edge_vector.unsqueeze(-1)), dim=1)
        self.dgl_g.ndata['features'] = self.features  # pass down features to dgl

    def step(self, action: int):  # make action and give reward
        done = False
        if self.edge_vector[action] == 1:  # if find neighbor = no reward (don't need node)
            '''
            right now, selecting na index twice doesnt do anything
            what if selecting an index twice removed the channel? increment budget, reward is negative change
            '''
            # reward = 0
            self.edge_vector[action] = 0  # mark channel for deletion
            self.take_action(action, remove=True)
            reward = 1.01 * self.get_reward()
        else:
            # reward = 0
            self.edge_vector[action] = 1  # mark as explored in edge vector
            self.take_action(action)
            reward = self.get_reward()

        if self.num_actions == self.budget + self.budget_offset:  # check if exhausted budget
            # self.get_reward()
            # reward = self.btwn_cent
            done = True
            self.r_logger.add_log('tot_reward', self.btwn_cent)
        info = {}
        reward = torch.Tensor([reward])
        return self.dgl_g, reward, done, info  # Tensor so we can take it and appending

    def get_illegal_actions(self):  # tells Ajay to not look at neighbors as an action
        illegal = (self.edge_vector == 1.).nonzero()  # if neighbor of node = illegal
        legal = (self.edge_vector == 0.).nonzero()
        return illegal, legal

    def get_reward(self):  # if add node, what is the betweeness centrality?
        new_btwn = self.ig_g.betweenness(self.node_id, weights=self.ig_g.es["weight"]) / self.norm
        # new_btwn = self.get_triangles()
        reward = new_btwn - self.btwn_cent  # how much improve between new & old btwn cent
        self.btwn_cent = new_btwn  # updating btwn cent to compare on next node
        return reward

    def get_triangles(self):
        triangles = 0
        incident_edges = [list(x.tuple) for x in self.ig_g.es.select(_source=[self.node_index])]
        if incident_edges:
            nbrs = np.unique(reduce(lambda x, y: x + y, incident_edges))
            nbrs = nbrs[nbrs != self.node_index]
            nbrsnbrs = self.ig_g.es.select(_between=(nbrs, nbrs))
            triangles = len(nbrsnbrs)
        return triangles

    def take_action(self, action, remove=False):
        neighbor_index = action
        neighbor_id = self.index_to_node[neighbor_index]
        if remove:
            self.ig_g.delete_edges((neighbor_id, self.node_id))
            self.features[action, -1] = 0
            self.num_actions -= 1
        else:
            self.ig_g.add_edge(neighbor_id, self.node_id, weight=1)
            self.features[action, -1] = 1
            self.num_actions += 1

        self.dgl_g.ndata['features'] = self.features

    def reset(self):
        if self.repeat and self.base_graph is not None:
            # reload
            self.nx_graph = deepcopy(self.base_graph)
        else:
            if self.graph_type == 'snapshot':
                self.nx_graph = get_random_snapshot()
                if self.node_id not in ["", None, self.k]:
                    self.index_to_node = bidict(enumerate(self.nx_graph.nodes()))
                    self.node_index = self.index_to_node.inverse[self.node_id]
                else:
                    self.add_node("")
            elif self.graph_type == 'sub_graph':
                self.nx_graph = get_random_snapshot()
                self.generate_subgraph()
                self.add_node("")
            elif self.graph_type == 'scale_free':
                self.nx_graph = random_scale_free(self.k)
                self.add_node(self.k)

            self.graph_size = len(self.nx_graph.nodes())

            # Create bidictionary = tuple index: pubKey
            self.index_to_node = bidict(enumerate(self.nx_graph.nodes()))

            if self.repeat:
                self.base_graph = deepcopy(self.nx_graph)

        # convert nx_graph for gcn and metrics
        self.ig_g = nx_to_ig(self.nx_graph)
        self.dgl_g = dgl.from_networkx(self.nx_graph.to_undirected()).add_self_loop()
        # self.dgl_g = dgl.from_networkx(self.nx_graph,edge_attrs=['weight']).add_self_loop()

        self.budget_offset = 0
        self.get_edge_vector_from_node()

        self.num_actions = 0

        self.get_features()

        weights = self.ig_g.es["weight"]
        self.norm = (self.graph_size * (self.graph_size - 1) / 2)
        self.w_norm = sum(weights)
        self.btwn_cent = self.ig_g.betweenness(self.node_id, weights=weights) / self.norm
        # self.btwn_cent = self.get_triangles()
        return self.dgl_g

    def draw_graph(self):
        if self.graph_type == 'snapshot':
            nx.draw(self.nx_graph,
                    node_color='blue',
                    edge_color='yellow',
                    node_size=150,
                    node_shape='.'
                    )
            plt.show()

        elif self.graph_type == 'sub_graph':
            nx.draw(self.nx_graph,
                    node_color='red',
                    edge_color='green',
                    node_size=150,
                    node_shape='h'
                    )
            plt.show()

        elif self.graph_type == 'scale_free':
            nx.draw(self.nx_graph,
                    node_color='purple',
                    edge_color='pink',
                    with_labels=True,
                    node_size=150,
                    node_shape='h'
                    )
            plt.show()

    def get_edge_vector_from_node(self):
        # Create a vector of zeros to the length of the graph_size
        self.edge_vector = torch.zeros(self.graph_size)

        if self.node_id is not None:
            # for edge in self.nx_graph.edges():  # trying to find neighbors of node
            #     if edge[0] == self.node_id:
            #         neighbor_index = self.index_to_node.inverse[edge[1]]
            #         self.edge_vector[neighbor_index] = 1
            #         self.budget_offset += 1  # update budget, discount if have neighbor
            incident_edges = [list(x.tuple) for x in self.ig_g.es.select(_source=[self.node_index])]
            if incident_edges:
                vertices = torch.Tensor(reduce(lambda x, y: x + y, incident_edges)).unique()
                vertices = vertices[vertices != self.node_index].type(torch.long)
                self.edge_vector = self.edge_vector.put(vertices, torch.ones(len(vertices)))
                self.budget_offset = len(vertices)

        return self.edge_vector

    def generate_subgraph(self):
        if self.k >= len(self.nx_graph):
            return self.nx_graph

        included_nodes = set()
        excluded_nodes = set(self.nx_graph.nodes())  # excluded= nodes already found
        unexplored_neighbors = set()  # empty set of unexplored neighbors = k doesnt allow to explore neighbor

        node = random.choice(list(excluded_nodes))  # Take a random node to start BFS=breadth first search
        excluded_nodes.difference_update([node])  # Take node from non-explored
        included_nodes.add(node)  # Add node in visited nodes
        while len(included_nodes) < self.k:
            neighbors = list(self.nx_graph.neighbors(node))  # Get all the neighbors from the node
            shuffle(neighbors)  # Make the nodes random
            # shorten neighbors so that subgraph stays below k
            cutoff = min(self.k - len(included_nodes), len(neighbors))  # If enough space, add n if not until is filled
            neighbors = set(neighbors[:cutoff])  # Get the nodes up to

            unexplored_neighbors = unexplored_neighbors.union(
                neighbors - included_nodes)  # Add cutoff = unexplored neighbors
            included_nodes = included_nodes.union(neighbors)  # Add return n into included nodes
            excluded_nodes.difference_update(neighbors)  # Take out the neighbors
            node = random.choice(list(unexplored_neighbors))  # Choose a random node from unexplored neighbors
            unexplored_neighbors.difference_update([node])  # Remove that node from unexplored
        self.nx_graph = nx.DiGraph(nx.subgraph(self.nx_graph, included_nodes))  # return networkx graph

    def add_node(self, node_id):
        self.node_id = node_id
        self.node_index = len(self.nx_graph)
        self.nx_graph.add_node(self.node_id)
