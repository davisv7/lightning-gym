from lightning_gym.utils import print_config, random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from baselines import *
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def gen_budget_data(start=2, stop=8, step=1):
    """
    Override budget in config and repeat the experiment from k=1->15
    :return:
    """
    # parser = argparse.ArgumentParser(description='Run  a simulation according to config.')
    # parser.add_argument("--config", type=str, default="configs/test_scale_free.conf")
    # args = parser.parse_args()
    config_loc = "configs/test_scale_free.conf"

    config = configparser.ConfigParser()
    config.read(config_loc)
    print_config(config)
    seed = config["env"].getint("seed", fallback=None)

    agent_results = []
    random_results = []
    between_results = []
    degree_results = []
    greedy_results = []
    trained_results = []

    if seed:
        random_seed(seed)
        print("seed set")
    env = NetworkEnvironment(config)
    for i in range(start, stop + 1, step):
        env.budget = i
        # env = NetworkEnvironment(config)
        agent = DiscreteActorCritic(env, config, test=True)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        greed = GreedyAgent(env)
        trained = TrainedGreedyAgent(env, config)

        agent_results.append(agent.test())
        random_results.append(rando.run_episode())
        between_results.append(topk_btwn.run_episode())
        degree_results.append(topk_degree.run_episode())
        greedy_results.append(greed.run_episode())
        trained_results.append(trained.run_episode())
        # print("Agent Results", agent_results[-1])
        # print("Random Results:", random_results[-1])
        # print("TopK Betweenness Results:", between_results[-1])
        # print("TopK Degree Results:", degree_results[-1])
        # print("Greed Results:", greedy_results[-1])
        # print("Trained Greed Results:", trained_results[-1])
        print(f"Testing budget {i} complete.")

    df = pd.DataFrame({"Random": random_results,
                       "Betweenness": between_results,
                       "Degree": degree_results,
                       "Agent": agent_results,
                       "Greedy": greedy_results,
                       "Trained Greedy": trained_results})
    df.plot().set_xticks(list(range(0, stop - start+1, step)), list(range(start, stop+1, step)))
    plt.title("Comparison of Betweenness Improvement")
    plt.xlabel("Budget")
    plt.ylabel("Betweenness Improvement")
    plt.show()


if __name__ == '__main__':
    gen_budget_data()
