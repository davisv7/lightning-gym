import networkx as nx
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
from ..Logger import Logger
from random import sample, shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import graph_tools as gt

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
        nk_g: networkit graph
        dgl_g: make DGL graph

        dyn_btwn_getter: the degree centrality
        btwn_cent: betweeness centrality
        edge_vector: neighbors of a node, (1=neighbor, 0=not neighbor)
        features: betweeness centrality, degree centrality, edge vector
        r_logger: keeps track of betweeness centrality & reward
        repeat:
        budget_offset:
        nx_graph: networkX graph
        params = kwards: pass keywords
        k: length of subgraph

        """
        self.budget = budget
        self.node_id = node_id
        self.node_index = None
        self.index_to_node = bidict()
        self.nk_g = None
        self.dgl_g = None
        self.graph_size = None
        self.dyn_btwn_getter = None
        self.btwn_cent = 0
        self.edge_vector = None
        self.features = None
        self.nodes = None
        self.edges = None
        self.r_logger = Logger()
        self.repeat = True
        self.budget_offset = 0
        self.nx_graph = None
        self.params = kwargs
        self.k = kwargs.get("k", None)
        self.repeat = kwargs.get("repeat", False)
        self.base_graph = None
        self.graph_type = kwargs.get('graph_type', 'sub_graph')
        assert self.graph_type in ['sub_graph', 'snapshot', 'scale_free'], 'You must use one of the following graph ' \
                                                                           'types, ' \
                                                                           'sub_graph, snapshot, or scale_free.'

    def print_configuration(self):
        print("Configurations:",
              "Graph Type: {}".format(self.graph_type),
              "Budget: {}".format(self.budget),
              "Repeat State: {}".format(self.repeat),
              sep="\n\t"
              )
        if self.node_id  is not None:
            print("Node Id:", self.node_id)



    def get_features(self):

        '''
        Initialize Algorithm
        Measures the extent to which a node lies on shortest
        paths between other nodes. Nodes with high betweeness
        could have considerable more influence.
        '''

        bc = nk.centrality.Betweenness(self.nk_g, normalized=True)  # sets algorithm to find betweeness centrality
        # bc = nx.betweenness_centrality(self.nx_graph,weight="weight",k=100)
        bc.run()  # Run the algorithm
        b_centralities = torch.Tensor(bc.scores()).unsqueeze(-1)  # makes list smaller size
        # print(bc.scores())

        '''Initialize Algorithm
        Indicates how close a node is to all other nodes in the network. 
        '''
        dc = nk.centrality.DegreeCentrality(self.nk_g, normalized=True)
        # Run the algorithm
        dc.run()
        d_centralities = torch.Tensor(dc.scores()).unsqueeze(-1)
        cc = nk.centrality.Closeness(self.nk_g, False, nk.centrality.ClosenessVariant.Generalized)
        cc.run()
        c_centralities = torch.Tensor(cc.scores()).unsqueeze(-1)

        ''' appending 3 features in a tensor
        '''
        self.features = torch.cat((b_centralities, d_centralities, c_centralities, torch.Tensor(self.edge_vector).unsqueeze(-1)), dim=1)
        self.dgl_g.ndata['features'] = self.features  # pass down features to dgl

    def update_features(self):
        col_idx = torch.Tensor(np.repeat(-1, self.features.size(0))).long()  # ????????????
        rows = torch.arange(self.features.size(0)).long()
        update_values = torch.FloatTensor(self.edge_vector)

        self.features[rows, col_idx] = update_values
        self.dgl_g.ndata['features'] = self.features  # .ndta??

    def step(self, action: int):  # make action and give reward
        done = False
        if self.edge_vector[action] == 1:  # if find neighbor = no reward (don't need node)
            reward = 0
        else:

            '''what if, by adding a node twice, we remove it? increase budget, reward is negative change
            we would have to reinitialize the betweenness calculator on edge removals, which isn't terrible
            '''
            self.edge_vector[action] = 1  # mark as explored in edge vector
            # calculate reward
            reward = self.get_reward(action)
            self.update_features()

        if sum(self.edge_vector) == self.budget + self.budget_offset:
            done = True
            self.r_logger.add_log('tot_reward', self.btwn_cent)  # check if exceeded budget
            print("{:.4f}".format(self.btwn_cent))

        info = {}

        return self.dgl_g, torch.Tensor([reward]), done, info  # Tensor so we can take it and appending

    def get_illegal_actions(self):  # tells Ajay to not look at neighbors as an action
        illegal = (self.edge_vector == 1.).nonzero()  # if neighbor of node = illegal
        legal = (self.edge_vector == 0.).nonzero()
        return illegal, legal

    def get_reward(self, action):  # if add node, what is the betweeness centrality?
        neighbor_index = action
        event_type = 3  # nk.dynamic.GraphEvent.EDGE_ADDITION = 3

        # add an edge in one direction
        self.nk_g.addEdge(self.node_index, neighbor_index, w=1)  # making action node our neighbor
        event = nk.dynamic.GraphEvent(event_type, self.node_index, neighbor_index, 1)  # Something happends to the graph
        self.dyn_btwn_getter.update(event)  # degree centrality

        # # and another in the other direction
        # self.nk_g.addEdge(neighbor_index, self.node_index, w=1)
        # event = nk.dynamic.GraphEvent(event_type, neighbor_index, self.node_index, 1)
        # self.dyn_btwn_getter.update(event)
        new_btwn = self.dyn_btwn_getter.getbcx() / (self.graph_size * (self.graph_size - 1) / 2)  # normalize Btwn Cent
        reward = new_btwn - self.btwn_cent  # how much improve between new & old btwn cent
        # Adding reward to logger
        # self.r_logger.add_logger(reward)
        self.btwn_cent = new_btwn  # updating btwn cent to compare on next node

        # reward = sum(nx.betweenness_centrality_source(self.nx_graph,sources=[""]).values())

        return reward

    def reset(self):

        # if repeat == true , take base graph = nxt graph that we load the first time

        if self.repeat and self.base_graph is not None:
            # reload
            self.nx_graph = deepcopy(self.base_graph)
        else:
            if self.graph_type == 'snapshot':
                self.get_random_snapshot()

            elif self.graph_type == 'sub_graph':
                self.get_random_snapshot()
                self.generate_subgraph()
                self.nx_graph.add_node("")

            elif self.graph_type == 'scale_free':
                self.random_scale_free()
                self.nx_graph.add_node(self.k)
            # print(len(self.nx_graph),len(self.nx_graph.edges))
            # plot_apsp(undirected(self.nx_graph))
            # self.draw_graph()  # Here we are drawing our graph

            self.graph_size = len(self.nx_graph.nodes())


            # Create bidictionary = tuple index: pubKey
            self.index_to_node = bidict(enumerate(self.nx_graph.nodes()))

            '''if want to test node_id, convert str key to int index '''
            if self.node_id is not None:
                self.node_index = self.index_to_node.inverse[self.node_id]
            else:  # ??????????????
                if self.graph_type == 'snapshot':
                    self.nx_graph.add_node("")
                    self.graph_size = len(self.nx_graph.nodes())

                self.node_index = self.graph_size-1
                # self.graph_size += 1
            if self.repeat:
                self.base_graph = deepcopy(self.nx_graph)
        self.gt_g = nx_to_gt(self.nx_graph)
        '''convert networkx to dgl'''
        self.dgl_g = dgl.from_networkx(self.nx_graph.to_undirected()).add_self_loop()

        self.budget_offset = 0
        self.get_edge_vector_from_node()
        # Passing network kit graph
        self.nk_g = nx_to_nk(self.nx_graph, self.index_to_node)

        self.get_features()

        '''get degree centrality of given node
        should we normalize the degree centrality????????
        '''
        self.dyn_btwn_getter = nk.centrality.DynBetweennessOneNode(self.nk_g, self.node_index)
        self.dyn_btwn_getter.run()
        # get betweeness centrality of given node, value is normalized
        self.btwn_cent = self.dyn_btwn_getter.getbcx() / (self.graph_size * (self.graph_size - 1) / 2)

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

    def get_random_snapshot(self):
        # make random graph
        randomfilename = get_random_filename()
        nodes, edges = load_json(path.join(SAMPLEDIRECTORY, randomfilename))
        self.nodes = nodes  # Added nodes
        self.edges = edges  # Added edges
        # Create nx_graph
        self.nx_graph = make_nx_graph(nodes, edges)

    # render on frame of environment at a time

    # Given public ID number from a node
    def get_edge_vector_from_node(self):
        # Create a vector of zeros to the length of the graph_size
        self.edge_vector = np.array([0 for _ in range(self.graph_size)])

        if self.node_id is not None:  # why double checking node id?????????
            for edge in self.edges:  # trying to find neighbors of node
                if edge[0] == self.node_id:
                    neighbor_index = self.index_to_node.inverse[edge[1]]
                    self.edge_vector[neighbor_index] = 1
                    self.budget_offset += 1  # update budget, discount if have neighbor

        return self.edge_vector

    def get_random_node_key(self):
        random_key = random.choice(self.nodes)
        return random_key

    # def render(self, mode='channel'):
    #     pass

    def generate_subgraph(self):
        if self.k > len(self.nx_graph):
            return self.nx_graph

        included_nodes = set()
        excluded_nodes = set(self.nx_graph.nodes())  # excluded= nodes already found
        unexplored_neighbors = set()  # empty set of unexplored neighbors = k doesnt allow to explore neighbor

        node = random.choice(list(excluded_nodes))  # Take a random node to start BFS=breathd first search
        excluded_nodes.difference_update([node])  # Take node from non-explored
        included_nodes.add(node)  # Add node in visited nodes
        while len(included_nodes) < self.k:
            neighbors = list(self.nx_graph.neighbors(node))  # Get all the neighbors from the node
            shuffle(neighbors)  # Make the nodes random
            # shorten neighbors so that subgraph stays below k
            cutoff = min(self.k - len(included_nodes), len(neighbors))  # If enough space, add n if not until is filled
            neighbors = set(neighbors[:cutoff])  # Get the nodes up to

            unexplored_neighbors = unexplored_neighbors.union(
                neighbors - included_nodes)  # Add cutoff = uexplored neighbors
            included_nodes = included_nodes.union(neighbors)  # Add return n into included nodes
            excluded_nodes.difference_update(neighbors)  # Take out the neighbors
            node = random.choice(list(unexplored_neighbors))  # Choose a random node from unexplored neighbors
            unexplored_neighbors.difference_update([node])  # Remove that node from unexplored
        self.nx_graph = nx.DiGraph(nx.subgraph(self.nx_graph, included_nodes))  # return networkx graph

    def random_scale_free(self):  # ?????????????????
        self.nx_graph = nx.scale_free_graph(self.k, 0.8, 0.1, 0.1).to_undirected()
