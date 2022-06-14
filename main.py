from graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config, random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.graph_utils import *
from baselines import *
import argparse


def create_snapshot_env(config):
    json_filename = config["env"]["filename"]
    ds = config.getboolean("env", "down_sample")
    nodes, edges = load_json(path.join(getcwd(), "snapshots", json_filename))
    key_to_alias = dict({x["pub_key"]: x["alias"] for x in nodes})
    size_before = len(nodes), len(edges)
    print(size_before)
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

    size_after = len(g.nodes()), len(g.edges()) // 2
    print(size_after)

    # create an environment, an agent, and then train for some number of episodes
    return NetworkEnvironment(config, g=g), key_to_alias, (size_before, size_after)


def main():
    """
    This program expects there to exist a directory containing snapshots of the lightning network stored in json format.
    Each of these files is loaded into a graph, and the nodes and edges are filtered according to some requirements.
    The filtered graph is then saved in another directory of 'clean' snapshots.
    A smaller directory is created that contains a sample of 100 of the clean snapshots to be used to train the agent.
    :return:
    """
    parser = argparse.ArgumentParser(description='Run  a simulation according to config.')
    # parser.add_argument("--config", type=str, default="configs/train_snapshot.conf")
    # parser.add_argument("--config", type=str, default="./configs/train_scale_free.conf")
    # parser.add_argument("--config", type=str, default="configs/test_snapshot.conf")
    parser.add_argument("--config", type=str, default="configs/test_scale_free.conf")
    args = parser.parse_args()
    config_loc = args.config

    config = configparser.ConfigParser()
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)

    if seed:
        random_seed(seed)
        print("seed set")

    if config["env"]["graph_type"] == "snapshot":
        env, k_to_a, _ = create_snapshot_env(config)
    else:
        env = NetworkEnvironment(config)

    ajay = DiscreteActorCritic(env, config)
    # rando = RandomAgent(env)
    # topk_btwn = TopBtwnAgent(env)
    # topk_degree = TopDegreeAgent(env)
    # greed = GreedyAgent(env)
    kCenter = kCenterAgent(env)
    # trained = TrainedGreedyAgent(env, config)

    # num_episodes = config.getint("training", "episodes")
    # for episode in range(num_episodes):
    #     log = ajay.train()
    #     recommendations = env.get_recommendations()
    #     print("E: {}, R: {:.4f}, N:{}".format(episode, env.btwn_cent, recommendations))
    # ajay.save_model()

    print("Test Results:", ajay.test())
    # print(ajay.problem.get_closeness())
    # print(ajay.problem.get_recommendations())
    # print([k_to_a[key] for key in ajay.problem.get_recommendations()])
    # # print("Random Results:", rando.run_episode())
    # print("TopK Results:", topk_btwn.run_episode())
    # print(topk_btwn.problem.get_closeness())
    # print(topk_btwn.problem.get_recommendations())
    # print([k_to_a[key] for key in topk_btwn.problem.get_recommendations()])
    # print("TopK Degree Results:", topk_degree.run_episode())
    # print("Trained Greedy Results:", trained.run_episode())
    print("kCenter Results:", kCenter.run_episode())
    # # print("Greed Results:", greed.run_episode())
    # print('total reward: ', ajay.logger.log['tot_reward'])
    # print("td error: ", ajay.logger.log['td_error'])
    # print("entropy: ", ajay.logger.log['entropy'])
    # ajay.logger.plot_reward()
    # ajay.logger.plot_td_error()


if __name__ == '__main__':
    main()
