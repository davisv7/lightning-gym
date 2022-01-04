import numpy as np
from lightning_gym.envs.lightning_network import NetworkEnvironment
import torch
import torch.nn.functional as F
from lightning_gym.GCN import GCN
from sklearn.preprocessing import MinMaxScaler


class TopBtwnAgent:
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
            _, _, done, _ = self.problem.step(action, no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
        return self.problem.btwn_cent

    def compute_betweenness(self):
        # cost = None
        cost = "cost"
        betweennesses = self.problem.ig_g.betweenness(directed=True, weights=cost, cutoff=20)
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
            _, _, done, _ = self.problem.step(action, no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
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
                _, _, _, _ = self.problem.step(action, no_calc=True)

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
            _, _, done, _ = self.problem.step(action.item(), no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
        return self.problem.btwn_cent

    def pick_random_action(self):
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        return np.random.choice(legal_actions)


class TrainedGreedyAgent:
    def __init__(self, problem: NetworkEnvironment, config):
        self.problem = problem  # environment
        self.path = config.get("agent", "model_file")
        self.in_feats = config.getint("agent", "in_features")
        self.hid_feats = config.getint("agent", "hid_features")
        self.out_feats = config.getint("agent", "out_features")
        self.layers = config.getint("agent", "layers")
        self.model = GCN(self.in_feats, self.hid_feats, self.out_feats, n_layers=self.layers, activation=F.rrelu)
        self.load_model()
        self.path = config.get("agent", "model_file")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()

    def run_episode(self):  # similar to epochs
        done = False
        G = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_greedy_action(G)

            # take action
            _, _, done, _ = self.problem.step(action, no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
        print(self.problem.btwn_cent, self.problem.get_recommendations())
        return self.problem.btwn_cent

    def pick_greedy_action(self, G):
        illegal_actions = self.problem.get_illegal_actions().squeeze().detach().numpy()
        best_action = None
        n = 5
        scaler = MinMaxScaler()

        costs = 1 / (np.array(self.problem.ig_g.es()["cost"]) + 1)
        costs = scaler.fit_transform(costs.reshape(-1, 1)).squeeze()
        # costs = 1 - costs
        costs = torch.Tensor(costs).unsqueeze(-1)
        # th_layer = torch.nn.Threshold(-0.001, 1)
        # costs = th_layer(costs)
        # costs = 1 + costs
        [pi, _] = self.model(G, w=costs)
        # [pi, _] = self.model(G)
        if self.problem.num_actions + self.problem.budget_offset < 2:
            best_action = self.predict_action(pi, illegal_actions, 1).item()
        else:
            best_reward = -1
            predicted_actions = list(set(map(lambda x: x.item(), self.predict_action(pi, illegal_actions, n))))
            for a in predicted_actions:
                _, reward, _, _ = self.problem.step(a)
                reward = reward.item()
                if reward > best_reward:
                    best_action = a
                    best_reward = reward
                _, _, _, _ = self.problem.step(a, no_calc=True)

        return best_action

    def predict_action(self, pi, illegal_actions, n=1):

        # make illegal actions impossible
        pi = pi.squeeze()
        pi[illegal_actions] = -float('Inf')

        # Calculate distribution from policy network
        pi = F.softmax(pi, dim=0)
        dist = torch.distributions.categorical.Categorical(pi)
        probs = dist.probs.detach().numpy()

        if n == 1:
            return probs.argmax()  # take the most likely action
        else:
            return dist.sample((n,))  # sample n actions


class TopDegreeAgent:
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
            _, _, done, _ = self.problem.step(action, no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
        return self.problem.btwn_cent

    def compute_degrees(self):
        betweennesses = self.problem.ig_g.degree()
        names = self.problem.ig_g.vs()["name"]
        actions = [self.problem.index_to_node.inverse[name] for name in names]
        self.action_to_btwn = dict(zip(actions, betweennesses))

    def pick_topk_action(self):
        self.compute_degrees()
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        best_action = max(legal_actions, key=lambda x: self.action_to_btwn[x])
        return best_action
