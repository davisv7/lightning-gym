from lightning_gym.envs.lightning_network import NetworkEnvironment
import warnings
from ActorCritic import DiscreteActorCritic
from lightning_gym.Logger import Logger
from lightning_gym.utils import plot_apsp
import configparser

warnings.filterwarnings("ignore")


def train_upwards(config):
    entire_log = Logger()
    start = 4
    end = start + 5
    num_episodes = 100
    for power in range(start, end):  # creating i amount of subgraphs and testing each one
        k = 2 ** power
        config["env"]["n"] = str(k)
        env = NetworkEnvironment(config)
        env.r_logger = entire_log
        ajay = DiscreteActorCritic(env, config)  # Awaken Ajay the Agent

        for episode in range(num_episodes):  # For each x in range of num_episodes, training x amount of Ajay
            log = ajay.train()
            recommendations = env.get_recommendations()
            print("E: {}, S: {}, R: {:.4f}, N:{}".format(episode, k, log.log['tot_reward'][-1], recommendations))
        entire_log = log
        config["agent"]["load_model"] = "True"
        ajay.save_model()  # Save model to reuse and continue to improve on it
        print()

    print('total reward: ', env.r_logger.log['tot_reward'])
    print("td error: ", env.r_logger.log['td_error'])
    print("entropy: ", env.r_logger.log['entropy'])
    env.r_logger.plot_reward()


def before_after(k=1000, load_model=True):
    env = NetworkEnvironment(config)
    ajay = DiscreteActorCritic(env, config)
    env.reset()
    plot_apsp(env.nx_graph)
    log = ajay.test()
    print("E: 1, S: {}, R: {:.2f}".format(k, log.log["tot_reward"][-1]))
    plot_apsp(env.nx_graph)


def print_config(config):
    for section in config.sections():
        print(section)
        for option in config[section]:
            print(f"\t{option} = {config.get(section, option)}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    print(config.read("./configs/train_scale_free.conf"))
    print_config(config)
    train_upwards(config)
    # before_after()
