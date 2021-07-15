import random
import gym
from gym import Env, logger, spaces
import numpy as np
import json
from pathlib import Path
from os import getcwd,path,listdir
from random import choice
import gym.spaces
import Discrete, Box

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
    import random
    import gym
    from gym import Env, logger, spaces
    import numpy as np
    import json
    from pathlib import Path
    from os import getcwd, path, listdir
    from random import choice

    np.random.seed(0)

    def load_json(json_filename):
        with open(json_filename, 'r') as json_file:
            # Pass json data as dictionary
            data = json.load(json_file)
            nodes = data['nodes']
            edges = data['edges']
        return nodes, edges

    # Environment Class
    class NetworkEnvironment(Env):

        def __init__(self, budget=10):
            # budget = number of channels that can add to graph
            # finding shortest path(fees)- why iterating through each node to accomodate budget
            # iterate though each node and keeps track of state, decreasing budget as go though iteration, once run out of budget= done --> move onto next one
            # use gcn to find most connected nodes of betweenus centrality
            # figuring out which nodes in network have the most influence
            self.budget = budget
            self.action_space = Discrete()
            self.observation_space = Box(low=np.array([0]), high=np.array([2]))
            self.state = []
            self.node_id = ()

        def step(self, action: int):
            self.state[action] = 1

            # calculate reward
            if self.state[action] == 1:
                reward = 1
            else:
                reward = -1
                done = True

            # check if done with budget
            if self.state[action] < self.budget:
                done = True,

            else:
                done = False
            return self.state[action], reward, done

        def get_reward():

            self.node_id
            # return < 1

        def reset(self):
            # gets random file name, loads graphs, returns state
            directory = getcwd()
            samplegraphdirectory = path.join(directory, 'sample_snapshots')
            graphfilenames = listdir(samplegraphdirectory)
            randomfilename = choice(graphfilenames)
            activeNodes, activeEdges = load_json(path.join(samplegraphdirectory, randomfilename))

            # Create graph
            G = nx.DiGraph()
            G.add_nodes_from(activeNodes)
            G.add_edges_from(activeEdges)

            graphsize = len(G.nodes())

            state = [0 for i in range(graphsize)]

            done = False

            return state, done

        ###visualize networkx
        # G = self.G
        # pos = nx.circular_layout(G)
        # nx.draw_networkx_nodes(G, pos, node_size=700, nodelist=self.nodes)
        # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
        # nx.draw_networkx_edges(G, pos, edgelist=self.edges,width=6)

        # nx.draw_networkx_nodes(G, pos, node_size=1400, nodelist=[self.state.state], node_color='red')
        # nx.draw_networkx_edge_labels(G,pos,edge_labels={('Se','Sn'):'Hi'},font_color='red')

        # plt.axis('off')
        # plt.show()

        # render on frame of environment at a time
        def render(self, mode='channel'):
            print('bandits success prob:')
            for i in range(self.num_bandits):
                print("channel {num} reward prob: {prob}".format(num=i, prob=self.bandit_success_prob[i]))