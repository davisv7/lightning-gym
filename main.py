# from lightning_network import *
from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings

warnings.filterwarnings("ignore")

budget = 100
env = NetworkEnvironment(budget=budget, node_id=None)
env.reset()

for i in range(budget):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(f"by taking adding node {action}, we increased our betweenness centrality by {reward:2f}")
