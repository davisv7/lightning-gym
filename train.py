from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
from lightning_gym.utils import plot_apsp
import configparser
from lightning_gym.utils import random_seed
from baselines import TrainedGreedyAgent

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
    for episode in range(config.getint("training", "episodes")):
        log = ajay.train()
        if pog:
            log.add_log("pog", get_pog(env, config, log.get_last_reward()))
        recommendations = env.get_recommendations()
        print("E: {}, S: {}, R: {:.4f}, N:{}".format(episode,
                                                     env.n,
                                                     log.get_last_reward(),
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


def before_after(k=1000, load_model=True):
    env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    env.reset()
    plot_apsp(env.nx_graph)
    log = ajay.test()
    print("E: 1, S: {}, R: {:.2f}".format(k, log.log["tot_reward"][-1]))
    plot_apsp(env.nx_graph)


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    print(config.read("./configs/train_scale_free.conf"))
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)
    if seed:
        random_seed(seed)
    # train_upwards(config)
    log = train_agent(config, pog=True)
    # before_after()
    log.plot_reward(reward_type="pog")
