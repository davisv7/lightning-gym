import networkx as nx
from os import path, getcwd
import igraph as ig
from .utils import get_random_filename
from graph_utils import load_json
import numpy as np
from copy import deepcopy


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
    nodes, edges = load_json(path.join(getcwd(), randomfilename))
    # Create nx_graph
    return make_nx_graph(nodes, edges)


def get_snapshot(filename):
    """
    Get a random graph filename, load it, and return it as an nx_graph type
    :return:
    """
    # make random graph
    nodes, edges = load_json(path.join(getcwd(), filename))
    # Create nx_graph
    return make_nx_graph(nodes, edges)


def random_scale_free(k):
    return nx.DiGraph(nx.scale_free_graph(k, 0.8, 0.1, 0.1))


def nx_to_ig(nx_graph, add_self_loop=True):
    """
    Given an nx_digraph, convert it to an igraph.
    :param add_self_loop:
    :param nx_graph:
    :return:
    """
    ig_g = ig.Graph()
    edge_list = []
    costs = []

    # add vertices to igraph
    for node in nx_graph.nodes():
        ig_g.add_vertex(name=node)

    # add normal edges to igraph
    for u, v in nx_graph.edges():
        c1 = float(nx_graph[u][v].get('cost', 0.1))
        edge_list.append((u, v))
        costs.append(c1)

    # add self loops (if applicable)
    if add_self_loop:
        max_cost = max(costs)
        for node in nx_graph.nodes():
            edge_list.append((node, node))
            costs.append(max_cost + 1)  # keeps these self loops from affecting betweenness algorithm
    ig_g.add_edges(edge_list, {'cost': costs})
    return ig_g


def down_sample(nx_graph, config):
    """
    Remove nodes randomly with respect to degree centrality until under n.
    :return:
    """
    n = config.getint("env", "n", fallback=None)
    node_id = config.get("env", "node_id")

    if len(nx_graph) <= n:
        return nx_graph

    new_nx_graph = deepcopy(nx_graph)
    degrees = np.array([y for x, y in nx_graph.degree()])
    probs = 1 - np.divide(degrees, sum(degrees))
    probs = np.divide(probs, sum(probs))
    nodes = nx_graph.nodes()
    un_chosen_ones = np.random.choice(nodes, len(nx_graph) - n, p=probs, replace=False)
    # un_chosen_ones = np.random.choice(nodes, len(nx_graph) - n,  replace=False)

    new_nx_graph.remove_nodes_from(un_chosen_ones)
    if node_id is not None:
        if node_id not in new_nx_graph.nodes():
            new_nx_graph.add_node(node_id)
            new_nx_graph = nx.subgraph(nx_graph, new_nx_graph.nodes())
    return new_nx_graph


def undirected(nx_graph):
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(nx_graph.nodes())
    seen = set()
    for u, v in nx_graph.edges():
        if (u, v) in seen or (v, u) in seen:
            continue
        else:
            seen.add((u, v))
        c1 = nx_graph[u][v].get('cost', 0.1)
        c2 = nx_graph[v][u].get('cost', 0.1)
        capacity = nx_graph[u][v].get('capacity')
        cost = max(c1, c2, 1)
        undirected_graph.add_edge(u, v, cost=cost, capacity=capacity)
    return undirected_graph
