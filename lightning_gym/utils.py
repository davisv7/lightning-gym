import matplotlib.pyplot as plt
import networkx as nx
from os import getcwd, path, listdir
from random import choice
import itertools


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
    plt.hist(shortest_lengths, bins=list(range(10)) + list(range(100, 1000, 100)) + list(range(1001, 10000, 1000)))
    plt.show()


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")
