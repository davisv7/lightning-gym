import numpy as np
from lightning_gym.envs.lightning_network import NetworkEnvironment
import torch
import torch.nn.functional as F
from lightning_gym.GCN import GCN
from sklearn.preprocessing import MinMaxScaler
import dgl


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
            _, _, done, _ = self.problem.step(action)  # Take action and find outputs
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
                self.problem.btwn_cent -= reward
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
    def __init__(self, problem: NetworkEnvironment, config, n=5):
        self.problem = problem  # environment
        self.path = config.get("agent", "model_file")
        self.in_feats = config.getint("agent", "in_features")
        self.hid_feats = config.getint("agent", "hid_features")
        self.out_feats = config.getint("agent", "out_features")
        self.layers = config.getint("agent", "layers")
        self.model = GCN(self.in_feats, self.hid_feats, self.out_feats, n_layers=self.layers, activation=F.rrelu)
        self.load_model()
        self.path = config.get("agent", "model_file")
        self.n = n

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
            no_calc = self.n == 1
            _, _, done, _ = self.problem.step(action, no_calc=no_calc)  # Take action and find outputs
        self.problem.get_reward()
        # print(self.problem.btwn_cent, self.problem.get_recommendations())
        return self.problem.btwn_cent

    def pick_greedy_action(self, G):
        illegal_actions = self.problem.get_illegal_actions().squeeze().detach().numpy()
        best_action = None
        scaler = MinMaxScaler()

        costs = 1 / (np.array(self.problem.ig_g.es()["cost"]) + 1)
        # costs = -np.array(self.problem.ig_g.es()["cost"])
        costs = scaler.fit_transform(costs.reshape(-1, 1)).squeeze()
        # costs = 1 - costs
        costs = torch.Tensor(costs).unsqueeze(-1)
        # th_layer = torch.nn.Threshold(-0.001, 1)
        # costs = th_layer(costs)
        # costs = 1 + costs
        [pi, _] = self.model(G, w=costs)
        # [pi, _] = self.model(G)
        if self.problem.num_actions + self.problem.budget_offset < 2 or self.n == 1:
            best_action = self.predict_action(pi, illegal_actions, 1).item()
        else:
            best_reward = -1
            predicted_actions = list(set(map(lambda x: x.item(), self.predict_action(pi, illegal_actions, self.n))))
            for a in predicted_actions:
                _, reward, _, _ = self.problem.step(a)
                reward = reward.item()
                self.problem.btwn_cent -= reward
                _, _, _, _ = self.problem.step(a, no_calc=True)
                if reward > best_reward:
                    best_action = a
                    best_reward = reward

        return best_action

    def predict_action(self, pi, illegal_actions, n=1):

        # make illegal actions impossible
        pi = pi.squeeze()
        pi[illegal_actions] = -float('Inf')

        # Calculate distribution from policy network
        PI = F.softmax(pi, dim=0)
        dist = torch.distributions.categorical.Categorical(PI)
        probs = dist.probs.detach().numpy()

        if n == 1:
            return probs.argmax()  # take the most likely action
        else:
            return np.argpartition(pi.detach().numpy(), -n)[-n:]  # take top n actions
            # TODO if there are less than n legal actions, the agent will suggest illegal actions


class TopDegreeAgent:
    def __init__(self, problem: NetworkEnvironment):
        self.problem = problem  # environment
        self.computed = False  # flag indicating whether betweennesses have been calculated
        self.action_to_degree = None

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
        degrees = self.problem.ig_g.degree()
        names = self.problem.ig_g.vs()["name"]
        actions = [self.problem.index_to_node.inverse[name] for name in names]
        self.action_to_degrees = dict(zip(actions, degrees))

    def pick_topk_action(self):
        self.compute_degrees()
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        best_action = max(legal_actions, key=lambda x: self.action_to_degrees[x])
        return best_action


class kCenterAgent:
    def __init__(self, problem: NetworkEnvironment):
        self.problem = problem  # environment
        self.computed = False  # flag indicating whether betweennesses have been calculated
        self.action_to_degrees = None
        self.heads = []
        self.clusters = []

    def run_episode(self):  # similar to epochs
        done = False
        _ = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_kcenter_action()

            # take action
            _, _, done, _ = self.problem.step(action, no_calc=True)  # Take action and find outputs
        self.problem.get_reward()
        return self.problem.btwn_cent

    def compute_degrees(self):
        degrees = self.problem.ig_g.degree()
        names = self.problem.ig_g.vs()["name"]
        actions = [self.problem.index_to_node.inverse[name] for name in names]
        self.action_to_degrees = dict(zip(actions, degrees))

    def pick_kcenter_action(self):
        # the joining node has no channels, open one up to the highest degree node
        # make it the head of the mega cluster
        if self.problem.num_actions + self.problem.budget_offset == 0:
            self.compute_degrees()
            legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
            best_action = max(legal_actions, key=lambda x: self.action_to_degrees[x])
            self.heads.append(best_action)
            self.clusters.append(self.problem.ig_g.vs["name"])
            self.clusters[0] = list(set(self.clusters[0]) - {best_action})
        else:
            if len(self.heads) == 0:
                print("Nodes with preexisting neighbors has not yet been implemented.")
                raise Exception
            # find the head of the next cluster,
            # the new head is the node whose distance from the head of its cluster is the greatest
            new_head = None
            max_distance = 0
            illegal_actions = self.problem.get_illegal_actions().squeeze().detach().numpy()
            for head, cluster in zip(self.heads, self.clusters):
                paths = self.problem.ig_g.get_shortest_paths(head, to=np.setdiff1d(cluster, illegal_actions))
                path_lengths = np.array(list(map(lambda x: len(x) - 1, paths)))
                path_lengths = np.where(path_lengths == -1, float("inf"), path_lengths)
                max_path_len = np.max(path_lengths)
                if max_path_len > max_distance:
                    max_ind = np.argmax(path_lengths, axis=0)
                    new_head = paths[max_ind][-1]
                    max_distance = max_path_len

            # if the distance of that node to its head is greater than its distance to the new head
            # add them to the new cluster
            new_cluster = []
            paths_to_new_head = self.problem.ig_g.get_shortest_paths(new_head)
            lengths_to_new_head = np.array(list(map(lambda x: len(x) - 1, paths_to_new_head)))
            lengths_to_new_head = np.where(lengths_to_new_head == -1, float("inf"), lengths_to_new_head)
            lengths_to_new_head = dict(zip(self.problem.ig_g.vs["name"], lengths_to_new_head))
            for head, cluster in zip(self.heads, self.clusters):
                paths = self.problem.ig_g.get_shortest_paths(head, to=cluster)
                lengths_to_curr_head = np.array(list(map(lambda x: len(x) - 1, paths)))
                lengths_to_curr_head = np.where(lengths_to_curr_head == -1, float("inf"), lengths_to_curr_head)
                lengths_to_curr_head = dict(zip(cluster, lengths_to_curr_head))
                for node, length_to_curr_head in lengths_to_curr_head.items():
                    length_to_new_head = lengths_to_new_head[node]
                    if length_to_curr_head >= length_to_new_head:
                        new_cluster.append(node)

            # remove nodes in the new cluster from the previous clusters
            for i in range(len(self.clusters)):
                self.clusters[i] = list(set(self.clusters[i]) - set(new_cluster))
            self.clusters.append(new_cluster)
            self.heads.append(new_head)
            best_action = new_head
        return best_action
