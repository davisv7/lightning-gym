from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config, random_seed
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.graph_utils import undirected, down_sample
from baselines import *
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def create_snapshot_env(config):
    json_filename = config["env"]["filename"]
    ds = config.getboolean("env", "down_sample")
    nodes, edges = load_json(path.join(getcwd(), "snapshots", json_filename))
    key_to_alias = dict({x["pub_key"]: x["alias"] for x in nodes})

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
    return NetworkEnvironment(config, g=g), key_to_alias


def main():
    """
    Override budget in config and repeat the experiment from k=1->15
    :return:
    """
    parser = argparse.ArgumentParser(description='Run  a simulation according to config.')
    parser.add_argument("--config", type=str, default="configs/test_scale_free.conf")
    args = parser.parse_args()
    config_loc = args.config

    config = configparser.ConfigParser()
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)

    agent_results = []
    random_results = []
    between_results = []
    degree_results = []
    greedy_results = []
    trained_results = []

    if seed:
        random_seed(seed)
        print("seed set")
    env = NetworkEnvironment(config)
    for i in range(2, 8):
        env.budget = i
        # env = NetworkEnvironment(config)
        agent = DiscreteActorCritic(env, config, test=True)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        greed = GreedyAgent(env)
        trained = TrainedGreedyAgent(env, config)

        agent_results.append(agent.test())
        print("Agent Results", agent_results[-1])
        random_results.append(rando.run_episode())
        print("Random Results:", random_results[-1])
        between_results.append(topk_btwn.run_episode())
        print("TopK Betweenness Results:", between_results[-1])
        degree_results.append(topk_degree.run_episode())
        print("TopK Degree Results:", degree_results[-1])
        greedy_results.append(greed.run_episode())
        print("Greed Results:", greedy_results[-1])
        trained_results.append(trained.run_episode())
        print("Trained Greed Results:", trained_results[-1])
        print()

    df = pd.DataFrame(
        {"Random": random_results,
         "Betweenness": between_results,
         "Degree": degree_results,
         "Agent": agent_results,
         "Greedy": greedy_results,
         "Trained Greedy": trained_results})
    df.plot()
    plt.title("Comparison of Betweenness Improvement")
    plt.xlabel("Budget")
    plt.ylabel("Betweenness Improvement")
    plt.xticks(list(range(0, 21)))
    plt.show()


if __name__ == '__main__':
    main()
