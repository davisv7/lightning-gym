from train import train_agent, print_config
from lightning_gym.utils import random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from baselines import *
from os.path import join
from main import create_snapshot_env
import pandas as pd
import matplotlib.pyplot as plt


def gen_monthly_data(config):
    month_prefixes = [
        "feb",
        "mar",
        "april",
        "may",
        "june",
        # "july",
        # "aug",
        # "sept",
        # "oct",
        # "nov",
        # "dec"
    ]

    verbose = config.getboolean("training", "verbose")

    agent_results = []
    random_results = []
    between_results = []
    degree_results = []
    # greedy_results = []
    # trained_results = []

    for month in month_prefixes:
        config["env"]["filename"] = f"{month}.json"
        config["env"]["repeat"] = "True"
        env, _ = create_snapshot_env(config)
        # print_config(config)

        # env = NetworkEnvironment(config)
        agent = DiscreteActorCritic(env, config, test=True)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        # greed = GreedyAgent(env)
        # trained = TrainedGreedyAgent(env, config)

        agent_results.append(agent.test())
        random_results.append(rando.run_episode())
        between_results.append(topk_btwn.run_episode())
        degree_results.append(topk_degree.run_episode())
        # greedy_results.append(greed.run_episode())
        # trained_results.append(trained.run_episode())
        if verbose:
            print(f"Testing month {month} complete.")

    df = pd.DataFrame({
        "Random": random_results,
        "Betweenness": between_results,
        "Degree": degree_results,
        "Agent": agent_results,
        # "Greedy": greedy_results,
        # "Trained Greedy": trained_results
    })
    df.to_pickle("monthly_data.pkl")
    df.plot().set_xticks(list(range(len(month_prefixes))), month_prefixes)
    plt.title("Comparison of Betweenness Improvement")
    plt.xlabel("Month")
    plt.ylabel("Betweenness Improvement")
    plt.show()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_loc = "./configs/test_snapshot.conf"
    config.read(config_loc)
    seed = config["env"].getint("seed", fallback=None)
    # print_config(config)
    if seed:
        random_seed(seed)
    gen_monthly_data(config)
