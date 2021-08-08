import networkx as nx
from os import path
import igraph as ig
from .utils import SAMPLEDIRECTORY, get_random_filename, load_json


def make_nx_graph(nodes, edges):
    """
    For each node in nodes, add a node with the pubkey as its ID
    Add all of the edges from the list as-is.
    :param nodes: list of nodes
    :param edges: list of edges
    :return:
    """
    nx_graph = nx.DiGraph()
    for node in nodes:
        nx_graph.add_node(node, id=node)
    nx_graph.add_edges_from(edges)
    return nx_graph


def get_random_snapshot():
    """
    Get a random graph filename, load it, and return it as an nx_graph type
    :return:
    """
    # make random graph
    randomfilename = get_random_filename()
    nodes, edges = load_json(path.join(SAMPLEDIRECTORY, randomfilename))
    # Create nx_graph
    return make_nx_graph(nodes, edges)


def random_scale_free(k):
    return nx.scale_free_graph(k, 0.8, 0.1, 0.1).to_undirected()


def nx_to_ig(nx_graph):
    """
    Given an nx_graph, convert it to an igraph.
    Nondirected, new fee is the min between
    :param nx_graph:
    :return:
    """
    ig_g = ig.Graph()
    for node in nx_graph.nodes():  # n nodes into the nk_graph
        ig_g.add_vertex(name=node)

    for u, v in nx_graph.edges():
        w1 = nx_graph[u][v].get('weight', 1)
        w2 = nx_graph[v][u].get('weight', 1)
        fee = max(min(w1, w2), 1)
        ig_g.add_edge(u, v, weight=fee)
    return ig_g


def undirected(nx_graph):
    seen = []
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(nx_graph.nodes())
    for u, v in nx_graph.edges():
        if (v, u) in seen:
            continue
        else:
            w1 = nx_graph[u][v].get('weight', 1)
            w2 = nx_graph[v][u].get('weight', 1)
            fee = max(w1, w2, 1)
            undirected_graph.add_edge(u, v, w=fee)
    return undirected_graph


def unweighted(nx_graph):
    seen = []
    unweighted_graph = nx.Graph()
    unweighted_graph.add_nodes_from(nx_graph.nodes())
    for u, v in nx_graph.edges():
        if (v, u) in seen:
            continue
        else:
            unweighted_graph.add_edge(u, v)
    return unweighted_graph
