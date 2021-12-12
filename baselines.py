import numpy as np
from lightning_gym.envs.lightning_network import NetworkEnvironment


class TopKAgent:
    def __init__(self, problem: NetworkEnvironment):
        self.problem = problem  # environment
        self.computed = False  # flag indicating whether betweennesses have been calculated
        self.action_to_btwn = None

    def run_episode(self):  # similar to epochs
        done = False
        _ = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_topk_action()

            # take action
            _, reward, done, _ = self.problem.step(action, test=True)  # Take action and find outputs

        return self.problem.btwn_cent

    def compute_betweenness(self):
        betweennesses = self.problem.ig_g.betweenness(directed=True, weights="cost")
        names = self.problem.ig_g.vs()["name"]
        actions = [self.problem.index_to_node.inverse[name] for name in names]
        self.action_to_btwn = dict(zip(actions, betweennesses))
        self.computed = True

    def pick_topk_action(self):
        if not self.computed:
            self.compute_betweenness()
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        best_action = max(legal_actions, key=lambda x: self.action_to_btwn[x])
        return best_action


class GreedyAgent:
    def __init__(self, problem: NetworkEnvironment):
        self.problem = problem  # environment

    def run_episode(self):  # similar to epochs
        done = False
        _ = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_greedy_action()

            # take action
            _, reward, done, _ = self.problem.step(action)  # Take action and find outputs

        return self.problem.btwn_cent

    def pick_greedy_action(self):
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        best_action = None
        if self.problem.num_actions + self.problem.budget_offset < 2:
            vs = self.problem.ig_g.vs()
            node_to_degree = {v["name"]: self.problem.ig_g.degree(v) for v in vs}
            legal_nodes = [self.problem.index_to_node[index] for index in legal_actions]
            best_node = max(legal_nodes, key=lambda x: node_to_degree[x])
            best_action = self.problem.index_to_node.inverse[best_node]
        else:
            best_reward = 0
            for action in legal_actions:
                _, reward, _, _ = self.problem.step(action)
                reward = reward.item()
                if reward > best_reward:
                    best_action = action
                    best_reward = reward
                _, _, _, _ = self.problem.step(action)

        return best_action


class RandomAgent:
    def __init__(self, problem):
        self.problem = problem  # environment

    def run_episode(self):  # similar to epochs
        done = False
        _ = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_random_action()

            # take action
            _, _, done, _ = self.problem.step(action.item(), test=True)  # Take action and find outputs

        return self.problem.btwn_cent

    def pick_random_action(self):
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        return np.random.choice(legal_actions)
