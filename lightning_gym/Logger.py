import matplotlib.pyplot as plt
from collections import defaultdict


class Logger:
    def __init__(self):
        # self.log = {'tot_reward': [], 'entropy': [], 'td_error': []}
        self.log = defaultdict(list)

    def add_log(self, log_type, val):
        self.log[log_type].append(val)

    def plot_reward(self):
        X = list(range(0, len(self.log['tot_reward'])))
        plt.plot(X, self.log['tot_reward'])
        plt.xlabel("Episodes")
        plt.ylabel("Betweeness Improvement")
        plt.title("Performance of RL Agent on MBIP")
        plt.show()

    def plot_td_error(self):
        X = list(range(0, len(self.log['td_error'])))
        plt.plot(X, self.log['td_error'])
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Performance of RL Agent on MBIP")
        plt.show()
