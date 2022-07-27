import random

from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
import configparser
from lightning_gym.utils import random_seed
from baselines import TrainedGreedyAgent, GreedyAgent

warnings.filterwarnings("ignore")


def get_pog(env, config, last_reward):
    old_setting = env.repeat
    env.repeat = True
    greedy = TrainedGreedyAgent(env, config)
    # greedy = GreedyAgent(env)
    g_reward = greedy.run_episode()
    env.repeat = old_setting
    return round(last_reward / g_reward, 4)


def train_agent(config, pog=False):
    env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    log = None
    verbose = config.getboolean("training", "verbose")
    for episode in range(config.getint("training", "episodes")):
        log = ajay.train()
        last_reward = ajay.problem.get_betweenness()
        if pog:
            pog = get_pog(env, config, last_reward)
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


def train_upwards(config, pog=True, pog_lim=128):
    log = Logger()
    start = 4
    end = start + 5
    for power in range(start, end):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        config["env"]["n"] = str(k)
        if pog_lim < k:
            pog = False
        log.extend_log(train_agent(config, pog=pog))
        config["agent"]["load_model"] = "True"
    return log


def train_single_multiple(config, pog=True):
    # repeatedly train on multiple graphs
    log = Logger()
    num_episodes = 25
    num_graphs = 10
    config["training"]["episodes"] = str(num_episodes)
    config["env"]["repeat"] = "True"
    for instance in range(num_graphs):
        log = train_agent(config, pog=pog)
        log.extend_log(train_agent(config, pog=pog))
        config["agent"]["load_model"] = "True"
        seed = random.randint(0, int(1e6))
        random_seed(seed)
        config["env"]["seed"] = str(seed)
    return log


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_loc = "./configs/train_scale_free.conf"
    # config_loc = "./configs/train_snapshot.conf"
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)
    if seed:
        random_seed(seed)
    pog = True
    # pog = False
    # log = train_upwards(config)
    log = train_agent(config, pog=pog)
    # log = train_single_multiple(config, pog=pog)
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
