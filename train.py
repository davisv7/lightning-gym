from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
from lightning_gym.utils import plot_apsp
import configparser
from lightning_gym.utils import random_seed
from baselines import TrainedGreedyAgent, kCenterAgent
import pandas as pd
import matplotlib.pyplot as plt
from lightning_gym.graph_utils import *

warnings.filterwarnings("ignore")


def get_greedy_reward(env, config):
    old_setting = env.repeat
    env.repeat = True
    greedy = TrainedGreedyAgent(env, config)
    _ = greedy.run_episode()
    g_reward = greedy.problem.get_betweenness()
    env.repeat = old_setting
    return g_reward


def train_agent(config, pog=False):
    if config["env"]["graph_type"] == "snapshot":
        g, k_to_a, _ = create_snapshot_env(config)
        env = NetworkEnvironment(config, g=g)
    else:
        env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    log = None
    verbose = config.getboolean("training", "verbose")
    for episode in range(config.getint("training", "episodes")):
        log = ajay.train()
        last_reward = ajay.problem.get_betweenness()
        if pog:
            g_reward = get_greedy_reward(env, config)
            pog = last_reward / g_reward
            log.add_log("pog", pog)
            last_reward = pog
        recommendations = env.get_recommendations()
        if verbose:
            print("E: {}, S: {}, R: {:.4f}, N:{}".format(episode,
                                                         env.n,
                                                         last_reward,
                                                         recommendations))
    ajay.save_model()  # Save model to reuse and continue to improve on it
    return log


def train_upwards(config):
    logs = []
    start = 4
    end = start + 5
    for power in range(start, end):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        config["env"]["n"] = str(k)
        log = train_agent(config)
        logs.append(log)
        config["agent"]["load_model"] = "True"
    return log


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_loc = "./configs/train_scale_free.conf"
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)
    if seed:
        random_seed(seed)
    pog = True
    # train_upwards(config)
    log = train_agent(config, pog=pog)
    # before_after()
    if pog:
        log.plot_reward(reward_type="pog")
    else:
        log.plot_reward()

    # config = configparser.ConfigParser()
    # config_loc = "./configs/test_scale_free.conf"
    # config.read(config_loc)
    # print_config(config)
    # seed = config["env"].getint("seed", fallback=None)
    # if seed:
    #     random_seed(seed)
    # before_after(config)
