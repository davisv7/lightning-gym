from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic

warnings.filterwarnings("ignore")

budget = 10
# node_id = "038f8302141b9b5e53d239578d8ee0699d4a3cb852f6e93ec43bdee7eebd115bef"
node_id = None
# Initial Budget

total_reward = 0

for i in range(5, 11):
    k = 2 ** i
    params = {
        "k": k
    }
    env = NetworkEnvironment(budget=budget, node_id=node_id, kwargs=params)  # Create class instance
    env.reset()
    agent = DiscreteActorCritic(env, cuda_flag=False, load_model=True)


    for i in range(1000):
        log = agent.train()

    agent.save_model()
    print()


# obs = env.reset()
# for i in range(budget):  # Iterate each int for our budget range
#     action, _states = model.predict(obs)
#     state, reward, done, _ = env.step(action)  # Make the envrioment take that step
#     total_reward += reward  # Update the reward
#     print('Action', action)
#     print(
#         f"by adding node {env.index_to_node[action]},"
#         f" our betweenness centrality by {reward:.5f} for a total of {total_reward}")
#     print(state.shape, '\n')
