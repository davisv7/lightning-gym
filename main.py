# from lightning_network import *
from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings

warnings.filterwarnings("ignore")

budget = 3
#Initial Budget
env = NetworkEnvironment(budget=budget, node_id=None)  #Create class instance
env.reset()  # Why reset?
total_reward = 0

random_key = env.get_random_node_key()
print(random_key)
edgesIds, edges_numbers, edge_vector  = env.get_edge_vector_from_node(random_key)
print(edgesIds, edges_numbers, edge_vector)

# print({env.index_to_node[random_key]})
# print(env.get_edge_vector_from_node(random_key))


# for i in range(budget): # Iterate each int for our budget range
#     action = env.action_space.sample() #Take a random action: An action is a node
#     state, reward, done, _ = env.step(action) #Make the envrioment take that step
#     total_reward += reward #Update the reward
#     print('Action', action)
#     print(
#         f"by adding node {env.index_to_node[action]}, our betweenness centrality by {reward:.5f} for a total of {total_reward}")
#     print(state.shape)
# #

