import os.path as path
from lightning_gym.utils import print_config, random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from lightning_gym.graph_utils import *
from baselines import *
import argparse
from collections import defaultdict

from lightning_gym.graph_utils import *


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
        g, k_to_a, _ = create_snapshot_env(config)
        env = NetworkEnvironment(config, g=g)
    else:
        env = NetworkEnvironment(config)
        k_to_a = defaultdict(str)

    ajay = DiscreteActorCritic(env, config)
    rando = RandomAgent(env)
    topk_btwn = TopBtwnAgent(env)
    topk_degree = TopDegreeAgent(env)
    kCenter = kCenterAgent(env)
    trained = TrainedGreedyAgent(env, config)
    # greed = GreedyAgent(env)

    # num_episodes = config.getint("training", "episodes")
    # for episode in range(num_episodes):
    #     log = ajay.train()
    #     recommendations = env.get_recommendations()
    #     print("E: {}, R: {:.4f}, N:{}".format(episode, env.btwn_cent, recommendations))
    # ajay.save_model()

    print("A2C Results:", ajay.test())
    print(ajay.problem.get_recommendations())
    print([k_to_a[key] for key in ajay.problem.get_recommendations()])
    print()

    print("Random Results:", rando.run_episode())
    print(rando.problem.get_recommendations())
    print([k_to_a[key] for key in rando.problem.get_recommendations()])
    print()

    print("TopK Btwn Results:", topk_btwn.run_episode())
    print(topk_btwn.problem.get_recommendations())
    print([k_to_a[key] for key in topk_btwn.problem.get_recommendations()])
    print()

    print("TopK Degree Results:", topk_degree.run_episode())
    print(topk_degree.problem.get_recommendations())
    print([k_to_a[key] for key in topk_degree.problem.get_recommendations()])
    print()

    print("kCenter Results:", kCenter.run_episode())
    print(kCenter.problem.get_recommendations())
    print([k_to_a[key] for key in kCenter.problem.get_recommendations()])
    print()

    print("Trained Greedy Results:", trained.run_episode())
    print(trained.problem.get_recommendations())
    print([k_to_a[key] for key in trained.problem.get_recommendations()])
    print()

    # print("Greed Results:", greed.run_episode())
    # print('total reward: ', ajay.logger.log['tot_reward'])
    # print("td error: ", ajay.logger.log['td_error'])
    # print("entropy: ", ajay.logger.log['entropy'])
    # ajay.logger.plot_reward()
    # ajay.logger.plot_td_error()


if __name__ == '__main__':
    main()
