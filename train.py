from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
from lightning_gym.utils import plot_apsp
warnings.filterwarnings("ignore")


def train_upwards(node_id=None, budget=10, num_episodes=200, load_model=False):
    entire_log = Logger()
    start = 6
    end = start + 7
    for power in range(start, end):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        env = NetworkEnvironment(
            budget=budget,
            node_id=node_id,
            k=k,
            repeat=False,  # Change to True or False
            graph_type='sub_graph'  # This can be changed to different graph types
        )
        print(env)
        env.r_logger = entire_log
        ajay = DiscreteActorCritic(
            env,
            cuda_flag=False,
            load_model=load_model,
        )  # Awaken Ajay the Agent

        for episode in range(num_episodes):  # For each x in range of num_episodes, training x amount of Ajay
            log = ajay.train()
            print("E: {}, S: {}, R: {:.4f}".format(episode, k, log.log['tot_reward'][-1]))
        entire_log = log
        load_model = True
        ajay.save_model()  # Save model to reuse and continue to improve on it
        print()

    print('total reward: ', env.r_logger.log['tot_reward'])
    print("td error: ", env.r_logger.log['td_error'])
    print("entropy: ", env.r_logger.log['entropy'])
    env.r_logger.plot_logger()


def before_after(node_id=None, budget=10, k=1000, load_model=True):
    env = NetworkEnvironment(
        budget=budget,
        node_id=node_id,
        k=k,
        repeat=True,  # Change to True or False
        graph_type='sub_graph'  # This can be changed to different graph types
    )

    ajay = DiscreteActorCritic(
        env,
        cuda_flag=False,
        load_model=load_model,
    )
    env.reset()
    plot_apsp(env.nx_graph)
    log = ajay.test()
    print("E: 1, S: {}, R: {:.2f}".format(k, log.log["tot_reward"][-1]))
    plot_apsp(env.nx_graph)



# def print_prompt():
#     print("The agent will run in the enviroment with the follwing paramaters:",
#           "budget: {}".format(values['budget']),
#           "node_id: {}".format(values['node_id']),
#           "num_episodes: {}".format(values['num_episodes']),
#           "load_model: {}".format(values['load_model']), sep="\n\t")


if __name__ == '__main__':
    # print_prompt()
    train_upwards()
    # before_after()
