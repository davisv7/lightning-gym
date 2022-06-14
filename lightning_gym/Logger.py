import matplotlib.pyplot as plt
from collections import defaultdict


class Logger:
    def __init__(self):
        # self.log = {'tot_reward': [], 'entropy': [], 'td_error': []}
        self.log = defaultdict(list)

    def add_log(self, log_type, val):
        self.log[log_type].append(val)

    def get_last_reward(self):
        return self.log["tot_reward"][-1]

    def plot_reward(self, reward_type="tot_reward"):
        X = list(range(0, len(self.log[reward_type])))
        plt.plot(X, self.log[reward_type])
        plt.xlabel("Episodes")
        if reward_type == "pog":
            ylabel = "Percentage of Greedy"
        else:
            ylabel = "Betweenness Improvement"
        plt.ylabel(ylabel)
        plt.title("Training Performance on Random Graphs")
        plt.show()

    def plot_td_error(self):
        X = list(range(0, len(self.log['td_error'])))
        plt.plot(X, self.log['td_error'])
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Performance of RL Agent on MBIP")
        plt.show()
