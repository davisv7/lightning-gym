from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
import configparser


def main():
    """
    This program expects there to exist a directory containing snapshots of the lightning network stored in json format.
    Each of these files is loaded into a graph, and the nodes and edges are filtered according to some requirements.
    The filtered graph is then saved in another directory of 'clean' snapshots.
    A smaller directory is created that contains a sample of 100 of the clean snapshots to be used to train the agent.
    :return:
    """
    config = configparser.ConfigParser()
    print(config.read("config.conf"))
    print_config(config)
    json_filename = config["env"]["filename"]
    nodes, edges = load_json(path.join(getcwd(), json_filename))
    # print(len(nodes), len(edges))
    # og_sizes = [len(nodes), len(edges)]
    # clean nodes
    active_nodes = get_pubkeys(nodes)

    # clean edges
    edge_filters = config["edge_filters"]
    active_edges = clean_edges(edges, edge_filters)
    active_edges = get_channels_with_fees(active_edges)
    # print(len(active_nodes), len(active_edges))

    # Create graph
    g = nx.MultiDiGraph()
    g.add_edges_from(active_edges)
    g = g.subgraph(active_nodes)
    # print(len(g.nodes), len(g.edges))

    # clean graph
    graph_filters = config["graph_filters"]
    if graph_filters.getboolean("combine_multiedges"):
        g = simplify_graph(g)
    # print(len(g.nodes), len(g.edges))
    if graph_filters.getboolean("remove_bridges"):
        g = nx.DiGraph(reduce_to_mainnet(g))  # removes bridges and their descendants, leaving a well connected graph
    # print(len(g.nodes), len(g.edges))
    # reductions = [1 - len(g.nodes) / og_sizes[0], 1 - len(g.edges) / og_sizes[1]]
    # print("Node Reductions: {:.2f}%, Edge Reduction: {:.2f}%".format(reductions[0] * 100, reductions[1] * 100))
    """
    at this point, we should have a graph with the following properties:
    - no multi-edges, in favor of whichever edge has the lowest weight
    - nodes whose degree is >= 2
    - nodes/edges whose deletion will not create a subgraph
    - non-disabled edges with some minimum capacity
    - edges that have defined policies
    """
    entire_log = Logger()
    num_episodes = 100
    env = NetworkEnvironment(config, g=g)
    env.r_logger = entire_log
    ajay = DiscreteActorCritic(env, config)  # Awaken Ajay the Agent

    for episode in range(num_episodes):  # For each x in range of num_episodes, training x amount of Ajay
        log = ajay.train()
        recommendations = env.get_recommendations()
        print("E: {}, R: {:.4f}, N:{}".format(episode, log.log['tot_reward'][-1], recommendations))
        ajay.save_model()  # Save model to reuse and continue to improve on it

    print('total reward: ', env.r_logger.log['tot_reward'])
    print("td error: ", env.r_logger.log['td_error'])
    print("entropy: ", env.r_logger.log['entropy'])
    env.r_logger.plot_logger()


if __name__ == '__main__':
    main()
