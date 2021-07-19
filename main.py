# from lightning_network import *
from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings

warnings.filterwarnings("ignore")

budget = 50
node_id = "038f8302141b9b5e53d239578d8ee0699d4a3cb852f6e93ec43bdee7eebd115bef"
# Initial Budget
env = NetworkEnvironment(budget=budget, node_id=node_id)  # Create class instance
env.reset()  # Why reset?
total_reward = 0

# for i in range(budget):  # Iterate each int for our budget range
#     action = env.action_space.sample()  # Take a random action: An action is a node
#     state, reward, done, _ = env.step(action)  # Make the envrioment take that step
#     total_reward += reward  # Update the reward
#     print('Action', action)
#     print(
#         f"by adding node {env.index_to_node[action]},"
#         f" our betweenness centrality by {reward:.5f} for a total of {total_reward}")
#     print(state.shape, '\n')
#
# env.r_logger.plot_logger()

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")
#
# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

