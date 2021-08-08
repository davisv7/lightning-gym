import json
import matplotlib.pyplot as plt
import networkx as nx
from os import getcwd, path, listdir
from random import choice
import itertools
import igraph as ig
''' 
    getcwd() returns the current working directory of a processes
    SAMPLEDIRECTORY - get pathname where the file sample_snapshots
    is located
'''
CWD = getcwd()
SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')


def get_random_filename():
    """
    Get a list of all the files names inside a given directory and randomly select a filename form the list.
    ONLY GETS THE FILENAME NOT THE FILE
    :return:
    """
    graphfilenames = listdir(SAMPLEDIRECTORY)
    randomfilename = choice(graphfilenames)
    return randomfilename


def load_json(json_filename):
    """
    Pass json_file and open file
    Load the data (dictionary). It has two keys: nodes and links
    For each node in nodes get the id and add them to single list.
    For each edge (key) in links add elements of the edge in a tuple.
    (source, target, {'weight' : value })
    :param json_filename: json file to load
    :returns
        NODES - List of ID's
        EDGES - List of tuples
    """
    with open(json_filename, 'r') as json_file:
        # Pass json data as dictionary
        data = json.load(json_file)
        nodes = [n["id"] for n in data['nodes']]
        edges = [(e["source"], e["target"], {"weight": e["weight"]}) for e in data['links']]
    return nodes, edges


def plot_apsp(nx_graph):  # apsp:all pair shortest paths
    shortest_lengths = list(nx.all_pairs_bellman_ford_path_length(nx_graph))  # gives list of lengths
    shortest_lengths = [x[1].values() for x in shortest_lengths]
    shortest_lengths = list(itertools.chain(*shortest_lengths))
    plt.hist(shortest_lengths, bins=list(range(10)) + list(range(100, 1000, 100)) + list(range(1001, 10000, 1000)))
    plt.show()
