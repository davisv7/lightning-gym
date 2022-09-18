from lightning_gym.utils import print_config, random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from baselines import *
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def gen_budget_data(config, start=1, stop=15, step=1):
    """
    Override budget in config and repeat the experiment from k=1->15
    :return:
    """
    verbose = config.getboolean("training", "verbose")

    agent_results = []
    random_results = []
    between_results = []
    degree_results = []
    greedy_results = []
    trained_results = []
    kcenter_results = []

    env = NetworkEnvironment(config)
    for i in range(start, stop + 1, step):
        env.budget = i
        # env = NetworkEnvironment(config)
        agent = TrainedGreedyAgent(env, config, n=1)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        # greed = GreedyAgent(env)
        trained = TrainedGreedyAgent(env, config)
        kcenter = kCenterAgent(env)

        agent_results.append(agent.run_episode())
        random_results.append(rando.run_episode())
        between_results.append(topk_btwn.run_episode())
        degree_results.append(topk_degree.run_episode())
        # greedy_results.append(greed.run_episode())
        trained_results.append(trained.run_episode())
        kcenter_results.append(kcenter.run_episode())
        print("Round:", i)
        print("A2C Results", agent_results[-1])
        print("Random Results:", random_results[-1])
        print("TopK Betweenness Results:", between_results[-1])
        print("TopK Degree Results:", degree_results[-1])
        # print("Greed Results:", greedy_results[-1])
        print("Trained Greedy Results:", trained_results[-1])
        print("kCenter Results:", kcenter_results[-1])
        print()
        if verbose:
            print(f"Testing budget {i} complete.")

    df = pd.DataFrame({
        # "Budget": list(range(start, stop + 1)),
        "Random": random_results,
        "Betweenness": between_results,
        "Degree": degree_results,
        "A2C": agent_results,
        "k-Center": kcenter_results,
        # "Greedy": greedy_results,
        "Trained Greedy": trained_results,
    }, index=list(range(0, stop - start + 1, step)))
    df.to_pickle("results/budget_data_4096.pkl")


def plot_changing_budget():
    df = pd.read_pickle("results/budget_data_4096.pkl")
    df = df.rename(columns={"Agent": "A2C", "kCenter": "k-Center"})
    df.plot()

    a = 15
    plt.xticks(np.arange(a), np.arange(1, a + 1))
    plt.title("Comparison of Betweenness Improvement (4096)")
    plt.xlabel("Budget")
    plt.ylabel("Betweenness Improvement")
    plt.show()


def before_after(config):
    env = NetworkEnvironment(config)
    ajay = TrainedGreedyAgent(env, config, n=1)

    done = False
    G = ajay.problem.reset()  # We get our initial state by resetting
    betweennesses = ajay.problem.get_betweennesses()

    while not done:  # While we haven't exceeded budget
        # Get action from policy network
        action = ajay.pick_greedy_action(G)
        # take action
        _, _, done, _ = ajay.problem.step(action)  # Take action and find outputs
        # record average path length

    new_betweennesses = ajay.problem.get_betweennesses()
    df = pd.DataFrame({
        "Betweennesses": sorted(betweennesses)[-10:],
        "Agent Diameter": sorted(new_betweennesses)[-10:],
        # "kCenter Avg": k_avg,
        # "kCenter Diameter": k_max
    })
    df.plot.hist()
    plt.ylim(ymin=0)  # this line
    plt.show()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_loc = "configs/test_scale_free.conf"
    config.read(config_loc)
    config["env"]["repeat"] = "True"
    seed = config["env"].getint("seed", fallback=None)
    print_config(config)
    if seed:
        random_seed(seed)
        print("seed set")
    gen_budget_data(config)
    plot_changing_budget()
    # before_after(config)
