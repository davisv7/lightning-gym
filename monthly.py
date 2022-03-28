from train import print_config
from lightning_gym.utils import random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from baselines import *
from main import create_snapshot_env
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def gen_monthly_data(config):
    verbose = config.getboolean("training", "verbose")

    agent_results = []
    random_results = []
    between_results = []
    degree_results = []
    # greedy_results = []
    # trained_results = []
    kcenter_results = []
    times = []

    sizes_before = []
    sizes_after = []

    month_prefixes = [
        "feb",
        "mar",
        "april",
        "may",
        "june",
        "july",
        "aug",
        "sept",
        "oct",
        "nov",
        "dec"
    ]
    for month in month_prefixes:
        config["env"]["filename"] = f"{month}.json"
        config["env"]["repeat"] = "True"
        env, _, (size_before, size_after) = create_snapshot_env(config)
        sizes_before.append(size_before)
        sizes_after.append(size_after)

        agent = TrainedGreedyAgent(env, config, n=1)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        # greed = GreedyAgent(env)
        # trained = TrainedGreedyAgent(env, config)
        kcenter = kCenterAgent(env)

        baseline_list = [agent,
                         rando,
                         topk_btwn,
                         topk_degree,
                         # greed,
                         # trained,
                         kcenter]
        results_list = [agent_results,
                        random_results,
                        between_results,
                        degree_results,
                        # greedy_results ,
                        # trained_results ,
                        kcenter_results]

        start = timer()
        round_time = []
        for baseline, r_list in zip(baseline_list, results_list):
            r_list.append(baseline.run_episode())
            round_time.append(timer() - (start+sum(round_time)))
        times.append(round_time)

        if verbose:
            print(f"Testing month {month} complete.")

    # performance
    df_baselines = pd.DataFrame({
        "Agent": agent_results,
        "Random": random_results,
        "Betweenness": between_results,
        "Degree": degree_results,
        # "Greedy": greedy_results,
        # "Trained Greedy": trained_results,
        "kCenter": kcenter_results
    }, index=month_prefixes)

    # network size
    nodes_b, edges_b = list(zip(*sizes_before))
    nodes_a, edges_a = list(zip(*sizes_after))
    df_sizes = pd.DataFrame({
        "Nodes Before": nodes_b,
        "Nodes After": nodes_a,
        "Edges Before": edges_b,
        "Edges After": edges_a
    }, index=month_prefixes)

    df_runtimes = pd.DataFrame(dict(zip(
        [
            "Agent",
            "Random",
            "Betweenness",
            "Degree",
            # "Greedy",
            # "Trained Greedy",
            "kCenter"
        ], zip(*times)
    )), index=month_prefixes)

    # save results
    df_baselines.to_pickle("monthly_data.pkl")
    df_sizes.to_pickle("sizes_data.pkl")
    # df_runtimes.to_pickle("runtime_data.pkl")


def plot_changing_month(df=None):
    if df is None:
        df = pd.read_pickle("monthly_data.pkl")
    df.plot()
    plt.title("Comparison of Betweenness Improvement")
    plt.xlabel("Month")
    plt.ylabel("Betweenness Improvement")
    plt.show()


def plot_network_size(df=None):
    if df is None:
        df = pd.read_pickle("sizes_data.pkl")
    df[['Nodes Before', 'Nodes After']].plot.bar(rot=0)
    plt.title("Change in Number of Nodes After Filtering")
    plt.xlabel("Month")
    plt.ylabel("Number of Nodes")
    plt.show()

    df[['Edges Before', 'Edges After']].plot.bar(rot=0)
    plt.title("Change in Number of Edges After Filtering")
    plt.xlabel("Month")
    plt.ylabel("Number of Edges")
    plt.show()


def plot_runtimes(df=None):
    if df is None:
        df = pd.read_pickle("runtime_data.pkl")
    print(df)
    df.plot()
    plt.title("Comparison of Runtimes")
    plt.xlabel("Month")
    plt.ylabel("Runtime (s)")
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
    plot_changing_month()
    plot_network_size()
    # plot_runtimes()
