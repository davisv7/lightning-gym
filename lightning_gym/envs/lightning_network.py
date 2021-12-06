from gym import Env
from bidict import bidict
import dgl
import torch
from ..Logger import Logger
from ..graph_utils import *
from os import getcwd, path

CWD = getcwd()
SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')


# Environment Class
class NetworkEnvironment(Env):

    def __init__(self, config, g=None):
        """
        Initializes NetworkEnvironment
        :param config: contains config variables related to the agent and environment
        :param g: optional argument to pass a graph instead of randomly generating one.
        """
        self.budget = config.getint("env", "budget")
        self.node_id = config.get("env", "node_id")
        self.repeat = config.getboolean("env", "repeat")
        self.graph_type = config.get("env", "graph_type")
        self.filename = config.get("env", "filename")
        self.cutoff = config.getint("env", "cutoff")
        self.n = config.getint("env", "n", fallback=None)
        self.config = config

        # graphs
        self.base_graph = g
        self.nx_graph = None
        self.ig_g = None
        self.dgl_g = None

        # useful attributes if node_id is not default
        self.budget_offset = 0
        self.action_mask = None
        self.preexisting_neighbors = None

        # features
        self.costs = None
        self.e_btwns = None
        self.node_features = None

        # related to game state
        self.node_vector = None
        self.btwn_cent = 0
        self.num_actions = 0

        self.node_index = None
        self.graph_size = None
        self.norm = None
        self.index_to_node = bidict()
        self.default_node_ids = ["", None, self.n]

        # make sure a valid graph type is being used
        valid_types = ['snapshot', 'random_snapshot', 'scale_free']
        assert self.graph_type in valid_types, "\nYou must use one of the following graphs types:\n\t{}".format(
            ',\n\t'.join(valid_types))

    def __str__(self):
        return \
            """Configurations:
              Budget: {}
              Node ID: {}
              Graph Type: {}
              Repeat State: {}""".format(self.budget, self.node_id, self.graph_type, self.repeat)

    def get_illegal_actions(self):
        """
        A node is illegal if the agent is already connected to that node, or if it is masked by the action mask
        :return:
        """
        illegal = ((self.node_vector + self.action_mask + self.preexisting_neighbors) > 0.).nonzero()
        return illegal

    def get_legal_actions(self):
        """
        A node is legal if the agent is not already connected to that node, and if it is not masked by the action mask
        :return:
        """
        legal = ((self.node_vector + self.action_mask + self.preexisting_neighbors) == 0.).nonzero()
        return legal

    def get_reward(self):
        """
        Reward here is defined as the change in betweenness centrality of node_id.
        Could also be represented by the betweenness of the most recently added edge.
        :return:
        """
        # new_btwn = self.ig_g.betweenness(self.node_id) / self.norm
        new_btwn = self.ig_g.betweenness(self.node_id, directed=True, weights="cost") / self.norm
        # new_btwn = nx.betweenness_centrality(self.nx_graph, weight="cost", seed=5785)[self.node_id]
        reward = new_btwn - self.btwn_cent  # how much improve between new & old btwn cent
        self.btwn_cent = new_btwn  # updating btwn cent to compare on next node
        # return reward
        return self.btwn_cent

    def get_recommendations(self):
        """
        Returns a sorted list of recommendations as indicated by actions_taken.
        :return:
        """
        actions_taken = (self.node_vector == 1).nonzero()
        return sorted([self.index_to_node[index.item()] for index in actions_taken])

    def step(self, action: int):
        """
        Update graph, node_vector, reward, and done using action
        :param action: index representing node being connected to
        :return:
        """
        done = False
        reward = 0
        if self.node_vector[action] == 1:  # if find neighbor = no reward (don't need node)
            '''
            right now, selecting na index twice doesnt do anything
            what if selecting an index twice removed the channel? increment budget, reward is negative change
            '''
            self.node_vector[action] = 0  # mark channel for deletion
            self.take_action(action, remove=True)
            # reward = 1.01 * self.get_reward()
        else:
            self.node_vector[action] = 1  # mark as explored in edge vector
            self.take_action(action)
            reward = self.get_reward()

        if self.num_actions == self.budget + self.budget_offset:  # check if budget has been exhausted
            # self.get_reward()
            # reward = self.btwn_cent
            done = True
        info = {}
        reward = torch.Tensor([reward])
        return self.dgl_g, reward, done, info

    def take_action(self, action, remove=False):
        """
        Actually take the action by adding or removing the edge between the action node and our node
        :param action: index representing node being connected to
        :param remove: boolean value indicating whether the edge should be added or removed
        :return:
        """
        neighbor_index = action
        neighbor_id = self.index_to_node[neighbor_index]
        if remove:
            self.ig_g.delete_edges((neighbor_id, self.node_id))
            self.ig_g.delete_edges((self.node_id, neighbor_id))
            self.node_features[action, -1] = 0
            self.num_actions -= 1
        else:
            self.ig_g.add_edge(neighbor_id, self.node_id, cost=0.1)
            self.ig_g.add_edge(self.node_id, neighbor_id, cost=0.1)
            self.dgl_g.add_edges([neighbor_index, self.node_index], [self.node_index, neighbor_index])
            self.costs = torch.cat([self.costs, torch.Tensor([[0.1], [0.1]])])
            self.node_features[action, -1] = 1
            self.num_actions += 1

        self.dgl_g.ndata['features'] = self.node_features

    def load_graph(self):
        if self.graph_type == 'snapshot':
            if self.base_graph is not None:
                self.nx_graph = deepcopy(self.base_graph)
            elif self.filename:
                self.nx_graph = get_snapshot(self.filename)

            if self.node_id not in self.default_node_ids:
                self.index_to_node = bidict(enumerate(sorted(self.nx_graph.nodes())))
                self.node_index = self.index_to_node.inverse[self.node_id]
            else:
                self.add_node("")
        elif self.graph_type == 'scale_free':
            self.nx_graph = random_scale_free(self.n)
            self.add_node(self.n)

        # Create bidictionary = tuple index: pubKey
        self.index_to_node = bidict(enumerate(sorted(self.nx_graph.nodes())))

        if self.repeat:
            self.base_graph = deepcopy(self.nx_graph)

    def reset(self):
        # reset graph
        if self.repeat and self.nx_graph is not None:
            self.nx_graph = deepcopy(self.base_graph)  # reload
        else:
            self.load_graph()

        # create dgl graph and igraph from nx_graph
        self.ig_g = nx_to_ig(self.nx_graph)
        self.dgl_g = dgl.from_networkx(self.nx_graph).add_self_loop()
        self.norm = (self.graph_size * (self.graph_size - 1))

        # reset edge attribute: cost
        self.costs = self.ig_g.es["cost"]
        cost_sum = np.sum(self.costs)
        cost_mean = cost_sum / len(self.costs)
        self.costs = self.costs / cost_mean
        self.costs = torch.Tensor(self.costs).unsqueeze(-1)

        self.budget_offset = 0
        self.node_vector = torch.zeros(self.graph_size)
        self.update_neighbor_vector()
        self.update_node_features()
        self.update_action_mask()

        self.num_actions = 0
        if self.budget_offset < 2:
            self.btwn_cent = 0
        else:
            self.btwn_cent = self.get_reward()
        return self.dgl_g

    def update_neighbor_vector(self):
        """
        Populate our preexisting neighbor vector with nodes our node is already adjacent to. Update the budget offset.
        :return:
        """
        self.preexisting_neighbors = torch.zeros(self.graph_size)
        # if self.preexisting_neighbors is None or not self.repeat:
        #     if self.node_id not in self.default_node_ids:
        #         incident_edges = [list(x.tuple) for x in self.ig_g.es.select(_source=[self.node_id])]
        #         if incident_edges:
        #             vertices = torch.Tensor(reduce(lambda x, y: x + y, incident_edges)).unique()
        #             vertices = vertices[vertices != self.node_index].type(torch.long)
        #             self.preexisting_neighbors = self.preexisting_neighbors.put(vertices, torch.ones(len(vertices)))
        #             self.budget_offset = len(vertices)

    def update_node_features(self):
        """
        Assigns to each node a feature vector containing including but not limited to:
        closeness centrality: the average cost to send funds to/from this node from/to other nodes in the network
        degree centrality: the fraction of edges incident to the node relative to the total number of edges
        load centrality: similar to betweenness centrality, except it only considers one cheapest path versus all
        strengths: sum of out weights divided the max of the sum of these out weights
        node vector: 0-1 vector indicating which nodes the agent is adjacent to
        :return:
        """

        if self.node_features is None or not self.repeat:
            self.node_features = self.node_vector.unsqueeze(-1)
        else:
            self.node_features = self.node_vector.unsqueeze(-1)
        self.dgl_g.ndata['features'] = self.node_features

    def update_action_mask(self):
        """
        Creates an action mask according to some constraints. Such constraints can include but are not limited to:
            minimum degree
            minimum average capacity per channel
            minimum reliability
            minimum edge betweenness
        If a node is in the action mask, it is deemed an illegal move and will not be taken by the agent.
        :return:
        """
        self.action_mask = np.zeros(self.graph_size)
        self.action_mask[self.index_to_node.inverse[self.node_id]] = 1  # make sure we cannot select our own node
        # if self.action_mask is None or not self.repeat:
        #     if self.graph_type == "scale_free":
        #         self.action_mask = np.zeros(self.graph_size)
        #     else:
        #         candidate_filters = self.config["action_mask"]
        #         self.action_mask = np.zeros(self.graph_size)
        #         min_degree = candidate_filters.getint("minimum_channels", 0)
        #         min_avg_cap = candidate_filters.getint("min_avg_capacity", 0)
        #         # min_reliability = candidate_filters.getfloat("min_reliability", None)
        #         # min_btwn = candidate_filters.getfloat("min_betweenness", 0)
        #
        #         for i, node in enumerate(self.nx_graph.nodes()):
        #             if node in self.default_node_ids:
        #                 continue
        #
        #             degree = self.nx_graph.degree(node)
        #             if degree < min_degree:
        #                 self.action_mask[i] = 1
        #                 continue
        #
        #             incident_capacities = [self.nx_graph[node][n]["capacity"] for n in self.nx_graph.neighbors(node)]
        #             total_capacity = sum(incident_capacities)
        #             avg_capacity = total_capacity / degree
        #
        #             if avg_capacity < min_avg_cap:
        #                 self.action_mask[i] = 1
        #                 continue
        #             # elif self.features[i][0] / self.norm == min_btwn:
        #             #     self.action_mask[i] = 1
        #             # if min_reliability:
        #             #     reliability = get_reliability(node)
        #             #     if reliability < min_reliability:
        #             #         action_mask[i] = 1
        #             #         continue

    def add_node(self, node_id):
        """
        Add node with node_id to the nx_graph, if it has not already been added.
        :param node_id:
        :return:
        """
        if node_id not in self.nx_graph.nodes():
            self.node_id = node_id
            self.node_index = len(self.nx_graph.nodes())
            self.nx_graph.add_node(self.node_id)
            self.graph_size = len(self.nx_graph.nodes())
