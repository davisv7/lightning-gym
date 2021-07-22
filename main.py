from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger

warnings.filterwarnings("ignore")

values = {
    'budget': 10,
    'node_id': None,
    'num_episodes': 1000,
    'load_model': False,
}


def train_upwards():
    budget = values['budget']
    # node_id = "038f8302141b9b5e53d239578d8ee0699d4a3cb852f6e93ec43bdee7eebd115bef"
    node_id = values['node_id']
    # Initial Budget

    total_reward = 0
    num_episodes = values['num_episodes']  # Change this back to 10k
    load_model = values['load_model']
    entire_log = Logger()
    start = 8
    end = start + 1
    for power in range(start, end):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        env = NetworkEnvironment(
            budget=budget,
            node_id=node_id,
            k=k,
            repeat=True,  # Change to True or False
            graph_type='sub_graph'  # This can be changed to different graph types
        )
        env.r_logger = entire_log
        ajay = DiscreteActorCritic(
            env,
            cuda_flag=False,
            load_model=load_model,
        )  # Awaken Ajay the Agent

        for episode in range(num_episodes):  # For each x in range of num_episodes, training x amount of Ajay
            log = ajay.train()
        entire_log = log
        load_model = True
        ajay.save_model()  # Save model to reuse and continue to improve on it
        print()

    print('total reward: ', env.r_logger.log['tot_reward'])
    print("td error: ", env.r_logger.log['td_error'])
    print("entropy: ", env.r_logger.log['entropy'])
    env.r_logger.plot_logger()


def print_prompt():
    print("The agent will run in the enviroment with the follwing paramaters:",
          "budget: {}".format(values['budget']),
          "node_id: {}".format(values['node_id']),
          "num_episodes: {}".format(values['num_episodes']),
          "load_model: {}".format(values['load_model']), sep="\n\t")


if __name__ == '__main__':
    print_prompt()
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
