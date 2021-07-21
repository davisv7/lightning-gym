from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic

warnings.filterwarnings("ignore")


def train_upwards():
    budget = 10
    # node_id = "038f8302141b9b5e53d239578d8ee0699d4a3cb852f6e93ec43bdee7eebd115bef"
    node_id = None
    # Initial Budget

    total_reward = 0
    num_episodes = 1000
    load_model = False
    for power in range(6, 7):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        #
        env = NetworkEnvironment(
            budget=budget,
            node_id=node_id,
            k=k,
            repeat=True,
            graph_type='scale_free' # this can be changed to different types
        )  # Create class instance
        ajay = DiscreteActorCritic(
            env,
            cuda_flag=False,
            load_model=load_model
        )  # activate ajay

        for episode in range(num_episodes):  # for each i in range of num_episodes, training i amount of ajay
            log = ajay.train()

        load_model = True
        ajay.save_model()  # save model to reuse and continue to improve on it
        print()

    print(env.r_logger.log)
    env.r_logger.plot_logger()


if __name__ == '__main__':
    train_upwards()

# warnings.filterwarnings("ignore")
#
# budget = 10
# # node_id = "038f8302141b9b5e53d239578d8ee0699d4a3cb852f6e93ec43bdee7eebd115bef"
# node_id = None
# # Initial Budget
#
# total_reward = 0
#
# for i in range(6, 7): #creating i amount of subgraphs and testing each one
#     k = 2 ** i
#     #
#     env = NetworkEnvironment(
#         budget=budget,
#         node_id=node_id,
#         # k=k,
#         repeat=True
#     )  # Create class instance
#     ajay = DiscreteActorCritic(env, cuda_flag=False, load_model=False)#activate ajay
#     for i in range(1000): #for each i in range of 10, training i amount of ajay
#         log = ajay.train()
#         # ajay.save_model()ds
#     ajay.save_model() #save model to reuse and continue to improve on it
#     print()
#
# print(env.r_logger.log)
# env.r_logger.plot_logger()
#     #
#     # ajay.save_model()
#     # print()
#
#
# # obs = env.reset()
# # for i in range(budget):  # Iterate each int for our budget range
# #     action, _states = model.predict(obs)
# #     state, reward, done, _ = env.step(action)  # Make the envrioment take that step
# #     total_reward += reward  # Update the reward
# #     print('Action', action)
# #     print(
# #         f"by adding node {env.index_to_node[action]},"
# #         f" our betweenness centrality by {reward:.5f} for a total of {total_reward}")
# #     print(state.shape, '\n')
