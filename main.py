from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config, random_seed
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.graph_utils import undirected, down_sample
from baselines import *


def main():
    """
    This program expects there to exist a directory containing snapshots of the lightning network stored in json format.
    Each of these files is loaded into a graph, and the nodes and edges are filtered according to some requirements.
    The filtered graph is then saved in another directory of 'clean' snapshots.
    A smaller directory is created that contains a sample of 100 of the clean snapshots to be used to train the agent.
    :return:
    """
    config = configparser.ConfigParser()
    # config.read("configs/train_snapshot.conf")
    config.read("configs/train_scale_free.conf")
    # config.read("configs/test_snapshot.conf")
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)

    if seed:
        print("seed set")
        random_seed(seed)

    if config["env"]["graph_type"] == "snapshot":
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
        env = NetworkEnvironment(config, g=g)
    else:
        env = NetworkEnvironment(config)

    # env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    rando = RandomAgent(env)
    greed = GreedyAgent(env)
    topk = TopKAgent(env)
    trained = TrainedGreedyAgent(env, config)

    num_episodes = config.getint("training", "episodes")
    for episode in range(num_episodes):
        log = ajay.train()
        recommendations = env.get_recommendations()
        print("E: {}, R: {:.4f}, N:{}".format(episode, env.btwn_cent, recommendations))
    ajay.save_model()

    ajay._test = True
    print("Test Results:", ajay.test())
    print(ajay.problem.get_recommendations())
    # print([key_to_alias[key] for key in ajay.problem.get_recommendations()])
    print("Random Results:", max([rando.run_episode() for _ in range(1)]))
    print("TopK Results:", topk.run_episode())
    # print("Trained Greedy Results:", trained.run_episode())
    # print("Greed Results:", greed.run_episode())
    print('total reward: ', ajay.logger.log['tot_reward'])
    print("td error: ", ajay.logger.log['td_error'])
    print("entropy: ", ajay.logger.log['entropy'])
    # ajay.logger.plot_reward()
    # ajay.logger.plot_td_error()


if __name__ == '__main__':
    main()
