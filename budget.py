from lightning_gym.utils import print_config, random_seed
from ActorCritic import DiscreteActorCritic
import configparser
from baselines import *
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Override budget in config and repeat the experiment from k=1->15
    :return:
    """
    parser = argparse.ArgumentParser(description='Run  a simulation according to config.')
    parser.add_argument("--config", type=str, default="configs/test_scale_free.conf")
    args = parser.parse_args()
    config_loc = args.config

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
    for i in range(2, 8):
        env.budget = i
        # env = NetworkEnvironment(config)
        agent = DiscreteActorCritic(env, config, test=True)
        rando = RandomAgent(env)
        topk_btwn = TopBtwnAgent(env)
        topk_degree = TopDegreeAgent(env)
        greed = GreedyAgent(env)
        trained = TrainedGreedyAgent(env, config)

        agent_results.append(agent.test())
        print("Agent Results", agent_results[-1])
        random_results.append(rando.run_episode())
        print("Random Results:", random_results[-1])
        between_results.append(topk_btwn.run_episode())
        print("TopK Betweenness Results:", between_results[-1])
        degree_results.append(topk_degree.run_episode())
        print("TopK Degree Results:", degree_results[-1])
        greedy_results.append(greed.run_episode())
        print("Greed Results:", greedy_results[-1])
        trained_results.append(trained.run_episode())
        print("Trained Greed Results:", trained_results[-1])
        print()

    df = pd.DataFrame(
        {"Random": random_results,
         "Betweenness": between_results,
         "Degree": degree_results,
         "Agent": agent_results,
         "Greedy": greedy_results,
         "Trained Greedy": trained_results})
    df.plot()
    plt.title("Comparison of Betweenness Improvement")
    plt.xlabel("Budget")
    plt.ylabel("Betweenness Improvement")
    plt.xticks(list(range(0, 21)))
    plt.show()


if __name__ == '__main__':
    main()
