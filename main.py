from ln_graph_utils.ln_graph_utils import *
import os.path as path
from os import getcwd
from lightning_gym.utils import print_config, random_seed
from lightning_gym.envs.lightning_network import NetworkEnvironment
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.graph_utils import undirected, down_sample,create_snapshot_env
from baselines import *
import argparse




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
    parser.add_argument("--config", type=str, default="configs/test_snapshot.conf")
    # parser.add_argument("--config", type=str, default="configs/test_scale_free.conf")
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
        env, k_to_a = create_snapshot_env(config)
    else:
        env = NetworkEnvironment(config)

    # env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    # rando = RandomAgent(env)
    # topk_btwn = TopBtwnAgent(env)
    # topk_degree = TopDegreeAgent(env)
    # greed = GreedyAgent(env)
    trained = TrainedGreedyAgent(env, config)

    num_episodes = config.getint("training", "episodes")
    for episode in range(num_episodes):
        log = ajay.train()
        recommendations = env.get_recommendations()
        print("E: {}, R: {:.4f}, N:{}".format(episode, env.btwn_cent, recommendations))
    ajay.save_model()

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
    print("Trained Greedy Results:", trained.run_episode())
    # # print("Greed Results:", greed.run_episode())
    # print('total reward: ', ajay.logger.log['tot_reward'])
    # print("td error: ", ajay.logger.log['td_error'])
    # print("entropy: ", ajay.logger.log['entropy'])
    # ajay.logger.plot_reward()
    # ajay.logger.plot_td_error()


if __name__ == '__main__':
    main()
