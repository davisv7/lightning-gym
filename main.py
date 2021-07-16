# from lightning_network import *
from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings

warnings.filterwarnings("ignore")

budget = 1
env = NetworkEnvironment(budget=budget, node_id=None)
env.reset()
total_reward = 0
for i in range(budget):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print(
        f"by adding node {env.index_to_node[action]}, our betweenness centrality by {reward:.5f} for a total of {total_reward}")
