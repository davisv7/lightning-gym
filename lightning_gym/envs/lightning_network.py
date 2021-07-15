from gym import Env, logger, spaces
from gym.spaces import Discrete, Box
import numpy as np
import json
import networkx as nx
from os import getcwd, path, listdir
from random import choice
import networkit as nk
from networkx import Graph as nx_Graph
from networkit import Graph as nk_Graph
from bidict import bidict
from typing import Dict
from ..utils import *

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
        self.nk_g = None
        self.graph_size = None
        self.dyn_btwn_getter = None
        self.btwn_cent = 0

    def step(self, action: int):
        done = False
        # calculate reward
        if self.state[action] == 1:
            reward = -float("inf")
            done = True
        else:
            self.state[action] = 1
            reward = self.get_reward(action)

        # check if done with budget
        if sum(self.state) == self.budget:
            done = True

        info = {}
        return self.state, reward, done, info

    def get_reward(self, action):
        neighbor_index = action
        event_type = 3  # nk.dynamic.GraphEvent.EDGE_ADDITION = 3

        #add an edge in one direction
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

        # Create nx_graph
        nx_graph = make_nx_graph(nodes, edges)

        self.graph_size = len(nx_graph.nodes())

        self.state = [0 for _ in range(self.graph_size)]

        self.index_to_node = bidict(enumerate(nx_graph.nodes()))
        self.nk_g = nx_to_nk(nx_graph, self.index_to_node)

        if self.node_id is None:
            # self.index_to_node[self.graph_size] = self.node_id
            self.node_index = self.graph_size
            self.nk_g.addNode()

        self.dyn_btwn_getter = nk.centrality.DynBetweennessOneNode(self.nk_g, self.node_index)
        self.dyn_btwn_getter.run()
        self.btwn_cent = self.dyn_btwn_getter.getbcx() / ((self.graph_size - 1) * (self.graph_size - 2) / 2)

        done = False
        self.action_space = Discrete(self.graph_size)

        return self.state, done

    # render on frame of environment at a time
    def render(self, mode='channel'):
        pass
