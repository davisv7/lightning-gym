from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
from lightning_gym.utils import plot_apsp
import configparser
from lightning_gym.utils import random_seed
from baselines import TrainedGreedyAgent
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_pog(env, config, last_reward):
    env.repeat = True
    greedy = TrainedGreedyAgent(env, config)
    g_reward = greedy.run_episode()
    env.repeat = False
    return round(last_reward / g_reward, 4)


def train_agent(config, pog=False):
    env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    log = None
    verbose = config.getboolean("training", "verbose")
    for episode in range(config.getint("training", "episodes")):
        log = ajay.train()
        last_reward = log.get_last_reward()
        if pog:
            pog = get_pog(env, config, log.get_last_reward())
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


def before_after(config, k=1024):
    env = NetworkEnvironment(config)
    ajay = TrainedGreedyAgent(env, config, n=1)

    # plot_apsp(env.nx_graph)
    # btwn_cent = ajay.run_episode()
    # print("E: 1, S: {}, R: {:.2f}".format(k, btwn_cent))
    # plot_apsp(env.nx_graph)
    done = False
    G = ajay.problem.reset()  # We get our initial state by resetting
    avg_path_lengths = []
    max_path_lengths = []
    while not done:  # While we haven't exceeded budget
        # Get action from policy network
        action = ajay.pick_greedy_action(G)

        # take action
        _, _, done, _ = ajay.problem.step(action)  # Take action and find outputs

        # record average path length
        avg_path_lengths.append(ajay.problem.ig_g.average_path_length())
        max_path_lengths.append(ajay.problem.ig_g.diameter())
    df = pd.DataFrame({
        "Average Path Lengths": avg_path_lengths,
        "Diameter": max_path_lengths
    })
    df.plot()
    plt.ylim(ymin=0)  # this line
    plt.show()


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

    config = configparser.ConfigParser()
    config_loc = "./configs/test_scale_free.conf"
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)
    if seed:
        random_seed(seed)
    before_after(config)
