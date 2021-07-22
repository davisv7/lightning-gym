import matplotlib.pyplot as plt
from collections import defaultdict

class Logger:
    def __init__(self):
        # self.log = {'tot_reward': [], 'entropy': [], 'td_error': []}
        self.log = defaultdict(list)
        self.plot_reward = None

    def add_log(self, log_type, val):
        self.log[log_type].append(val)


    def plot_logger(self):
        # y = len(self.log)
        X = list(range (0, len(self.log['tot_reward'])))
        plt.plot(X, self.log['tot_reward'])
        plt.show()
