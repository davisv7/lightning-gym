import matplotlib.pyplot as plt
import networkx as nx
from os import getcwd, path, listdir
from random import choice
import itertools
import random
import numpy as np
import torch
import dgl
import igraph


def get_random_filename():
    """
    Get a list of all the files names inside a given directory and randomly select a filename form the list.
    ONLY GETS THE FILENAME NOT THE FILE
    :return:
    """
    CWD = getcwd()
    SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')
    graphfilenames = listdir(SAMPLEDIRECTORY)
    randomfilename = choice(graphfilenames)
    return randomfilename


def plot_apsp(nx_graph):  # apsp:all pair shortest paths
    shortest_lengths = list(nx.all_pairs_bellman_ford_path_length(nx_graph))  # gives list of lengths
    shortest_lengths = [x[1].values() for x in shortest_lengths]
    shortest_lengths = list(itertools.chain(*shortest_lengths))
    plt.hist(shortest_lengths)
    plt.show()


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")


def random_seed(seed_value, use_cuda=False):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    torch.use_deterministic_algorithms(True)
    random.seed(seed_value)  # Python
    dgl.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
