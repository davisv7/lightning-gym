import json
import networkx as nx
from os import getcwd, path, listdir
from random import choice
from networkx import Graph as nx_Graph
from networkit import Graph as nk_Graph
from typing import Dict

''' 
    getcwd() returs the current working direcetory of a processs
    SAMPLEDIRECTORY - get pathname where the file sample_snapshots
    is located
'''
CWD = getcwd()
SAMPLEDIRECTORY = path.join(CWD, 'sample_snapshots')

'''
    listdir() - Return a list of all the files names inside a given directory
    choice() - Returns  a random selected file name form the graphfilenames
    ONLY GETS THE    FILE NAME NOT THE FILE
'''


def get_random_filename():
    graphfilenames = listdir(SAMPLEDIRECTORY)
    randomfilename = choice(graphfilenames)
    return randomfilename


'''
    Pass json_file and open file
    Load the data (dictionary). It has two keys: nodes and links
    The values is made up of dictionaries within dictionaries. 
    For each node in nodes get the id and add them to single list.
    For each edge (key) in links add elements of the edge in a tupple.
    (source, target, dict- {'weight' : value }
    NODES - List of ID's
    EDGES - List of tuples
'''


def load_json(json_filename):
    with open(json_filename, 'r') as json_file:
        # Pass json data as dictionary
        data = json.load(json_file)
        nodes = [n["id"] for n in data['nodes']]
        edges = [(e["source"], e["target"], {"weight": e["weight"]}) for e in data['links']]
    return nodes, edges


'''
    Pass nodes and edges
    
    For each node in nodes. Add node
    Add edges: From => to and add weight
'''


def make_nx_graph(nodes, edges):
    nx_graph = nx.DiGraph()
    for node in nodes:
        nx_graph.add_node(node, id=node)
    nx_graph.add_edges_from(edges)
    return nx_graph


def nx_to_nk(nx_graph: nx_Graph, index_to_node) -> (nk_Graph, Dict):
    """
    Given a NetworkX graph, converts it to a Networkit graph, and returns it.
    :param nx_graph: NetworkX type graph
    :return: nk_graph: Networkit type graph
    :return: node_ids: mapping of indices to pub_keys, useless if generated graph
    """
    ids_to_index = index_to_node.inverse  # Pass the bidict
    nk_graph = nk_Graph(weighted=True, directed=False)  # Intance of nk_graph

    # add nodes
    for node in nx_graph.nodes():  # n nodes into the nk_graph, ? i
        nk_graph.addNode()

    # add edges
    seen = []
    for u, v in nx_graph.edges():
        if (v, u) in seen:
            continue
        else:
            nk_graph.addEdge(ids_to_index[u], ids_to_index[v])  # Souce and target
            w1 = nx_graph[u][v]["weight"]
            w2 = nx_graph[v][u]["weight"]
            fee = min(w1, w2)
            nk_graph.setWeight(ids_to_index[u], ids_to_index[v], fee)
            seen.append((u, v))

    return nk_graph
