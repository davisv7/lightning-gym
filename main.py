from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
import configparser
import random
import numpy as np
import torch


def main():
    """
    This program expects there to exist a directory containing snapshots of the lightning network stored in json format.
    Each of these files is loaded into a graph, and the nodes and edges are filtered according to some requirements.
    The filtered graph is then saved in another directory of 'clean' snapshots.
    A smaller directory is created that contains a sample of 100 of the clean snapshots to be used to train the agent.
    :return:
    """
    config = configparser.ConfigParser()
    config.read("config.conf")
    print_config(config)
    json_filename = config["env"]["filename"]
    seed = config["env"].getint("seed", fallback=None)
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    nodes, edges = load_json(path.join(getcwd(), json_filename))

    # clean nodes
    active_nodes = get_pubkeys(nodes)

    # clean edges
    edge_filters = config["edge_filters"]
    active_edges = clean_edges(edges, edge_filters)
    active_edges = get_channels_with_attrs(active_edges)

    # Create graph
    g = nx.MultiDiGraph()
    g.add_edges_from(active_edges)
    g = nx.MultiDiGraph(g.subgraph(active_nodes))

    # clean graph
    graph_filters = config["graph_filters"]
    if graph_filters.getboolean("combine_multiedges"):
        g = simplify_graph(g)
    if graph_filters.getboolean("remove_bridges"):
        g = nx.DiGraph(reduce_to_mainnet(g))

    """
    at this point, we should have a graph with the following properties:
    - no multi-edges, in favor of whichever edge has the highest cost, capacities are combined
    - nodes whose degree is >= 2
    - nodes/edges whose deletion will not create a subgraph
    - nodes whose adjacent edges all have cycles
    - non-disabled edges with some minimum capacity
    - edges that have defined policies
    """

    # create an environment, an agent, and then train for some number of episodes
    env = NetworkEnvironment(config, g=g)
    # env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)

    num_episodes = config.getint("training", "episodes")
    for episode in range(num_episodes):
        log = ajay.train()
        recommendations = env.get_recommendations()
        print("E: {}, R: {:.4f}, N:{}".format(episode, log.log['tot_reward'][-1], recommendations))
        ajay.save_model()

    print('total reward: ', env.r_logger.log['tot_reward'])
    print("td error: ", env.r_logger.log['td_error'])
    print("entropy: ", env.r_logger.log['entropy'])
    env.r_logger.plot_logger()


if __name__ == '__main__':
    main()
