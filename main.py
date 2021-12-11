from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.utils import random_seed
from lightning_gym.graph_utils import undirected, down_sample
from baselines import RandomAgent, ErsoyAgent


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
    ds = config.getboolean("env", "down_sample")

    if seed:
        print("seed set")
        random_seed(seed)

    if config["env"]["graph_type"] == "snapshot":
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
        if ds:
            g = down_sample(g, config)

        # reduce graph
        graph_filters = config["graph_filters"]
        if graph_filters.getboolean("combine_multiedges"):
            g = simplify_graph(g)
        if graph_filters.getboolean("remove_bridges"):
            g = nx.DiGraph(reduce_to_mainnet(g))
        if graph_filters.getboolean("undirected"):
            g = undirected(g)
        if graph_filters.getboolean("unweighted"):
            nx.set_edge_attributes(g, values=0.1, name='cost')

        print(len(g.nodes()), len(g.edges()))

        # create an environment, an agent, and then train for some number of episodes
        env = NetworkEnvironment(config, g=g)
    else:
        env = NetworkEnvironment(config)

    # env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    rando = RandomAgent(env)
    soy = ErsoyAgent(env)

    num_episodes = config.getint("training", "episodes")
    for episode in range(num_episodes):
        log = ajay.train()
        recommendations = env.get_recommendations()
        print("E: {}, R: {:.4f}, N:{}".format(episode, log.log['tot_reward'][-1], recommendations))
    ajay.save_model()

    ajay._test = True
    print("Test Results:", ajay.test())
    print("Random Results:", rando.run_episode())
    print("Esroy Results:", soy.run_episode())
    print('total reward: ', ajay.logger.log['tot_reward'])
    print("td error: ", ajay.logger.log['td_error'])
    print("entropy: ", ajay.logger.log['entropy'])
    # ajay.logger.plot_reward()
    # ajay.logger.plot_td_error()


if __name__ == '__main__':
    main()
