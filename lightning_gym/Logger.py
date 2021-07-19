import matplotlib.pyplot as plt

class Logger():
    def __init__(self):
        self.log = []
        self.plot_reward = None

    def add_logger(self, reward):
        self.log.append(reward)

    def plot_logger(self):
        y = len(self.log)
        plt.plot(self.log)
        plt.show()
