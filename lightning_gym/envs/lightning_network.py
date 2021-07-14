import random
import gym
from gym import Env, logger, spaces
import numpy as np
import json
from pathlib import Path
from os import getcwd,path,listdir
from random import choice

np.random.seed(0)

def load_json(json_filename):
    with open(json_filename, 'r') as json_file:
      #Pass json data as dictionary
      data = json.load(json_file)
      nodes = data['nodes']
      edges = data['edges']
    return nodes,edges

# Environment Class
class NetworkEnvironment(Env):

    def __init__(self, budget=10):
        # budget = number of channels that can add to graph
        self.budget = budget
        self.action_space = Discrete()
        self.observation_space = Box(low=np.array([0]), high=np.array([2]))
        self.state = []

    def step(self, action):
        reward = 0
        done = True
        result_prob = np.random.random()
        if result_prob < self.bandit_success_prob[action]:
            reward = 1
        else:
            reward = 0

        info = {}

        return [0], reward, done, self.info

    def reset(self):
        #gets random file name, loads graphs, returns state
        directory = getcwd()
        samplegraphdirectory = path.join(directory, 'sample_snapshots')
        graphfilenames = listdir(samplegraphdirectory)
        randomfilename = choice(graphfilenames)
        activeNodes, activeEdges = load_json(path.join(samplegraphdirectory,randomfilename))

        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(activeNodes)
        G.add_edges_from(activeEdges)

        graphsize = len(G.nodes())

        state = [0 for i in range(graphsize)]

        done = False

        return state, done

    # render on frame of environment at a time
    def render(self, mode='channel'):
        print('bandits success prob:')
        for i in range(self.num_bandits):
            print("channel {num} reward prob: {prob}".format(num=i, prob=self.bandit_success_prob[i]))