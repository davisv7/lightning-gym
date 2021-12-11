from lightning_gym.GCN import GCN
from lightning_gym.EGNNC import EGNNC
import torch
import torch.nn.functional as F
from lightning_gym.Logger import Logger
import numpy as np


class DiscreteActorCritic:
    def __init__(self, problem, config, **kwargs):

        self.problem = problem  # environment
        self.path = config.get("agent", "model_file")
        self.cuda = config.getboolean("agent", "cuda")
        self._load_model = config.getboolean("agent", "load_model")
        # self.memory_replay_buffer = ReplayMemory(5000)

        # hyperparameters
        self.in_feats = config.getint("agent", "in_features")
        self.hid_feats = config.getint("agent", "hid_features")
        self.out_feats = config.getint("agent", "out_features")
        self.gamma = config.getfloat("agent", "gamma")
        self.layers = config.getint("agent", "layers")
        self.learning_rate = config.getfloat("agent", "learning_rate")
        self.num_episodes = 1
        self._test = kwargs.get("test", False)
        self.logger = Logger()

        # models
        self.model = GCN(self.in_feats, self.hid_feats, self.out_feats, n_layers=self.layers, activation=F.rrelu)

        if self._load_model:  # making model
            self.load_model()
        if self.cuda:
            self.model = self.model.cuda()

        self.gcn_optimizer = torch.optim.Adam([{
            "params": self.model.parameters(),
            "lr": self.learning_rate
        }])
        self.lr_schedule = torch.optim.lr_scheduler.StepLR(self.gcn_optimizer, step_size=1, gamma=0.999)

    def print_actor_configuration(self):
        print("\tLoad model: {}".format(self._load_model),
              "Learning Rate: {}".format(self.learning_rate),
              sep="\n\t")

    def run_episode(self):  # similar to epochs
        done = False
        G = self.problem.reset()  # We get our initial state by resetting
        # Add our illegal actions which are ones in the edge vector
        illegal_actions = self.problem.get_illegal_actions()  # getting neighbors
        # Initialize the list below
        PI = torch.empty(0)  # policy network
        R = torch.empty(0)  # reward
        V = torch.empty(0)  # value network
        old_state = None

        while not done:  # While we haven't exceeded budget
            # Use this if we have an NVIDIA graphics card (we don't)
            if self.cuda:
                G.ndata['features'] = G.ndata['features'].cuda()

            # convolve our graph
            # costs = np.array(self.problem.ig_g.es()["cost"])
            # max_cost = np.max(costs)
            # costs = 1-(costs / max_cost)
            # costs = torch.Tensor(costs).unsqueeze(-1)
            # th_layer = torch.nn.Threshold(-0.99, 0)
            # costs = th_layer(costs)
            # costs = -costs+1
            # [pi, val] = self.model(G, w=costs)
            [pi, val] = self.model(G)

            # Get action from policy network
            action = self.predict_action(pi, illegal_actions)

            # take action
            G, reward, done, _ = self.problem.step(action.item(), test=self._test)  # Take action and find outputs
            illegal_actions = self.problem.get_illegal_actions()

            # collect outputs of networks for learning - cat = appending for tensors
            # Probability for taking action
            PI = torch.cat([PI, pi[action].unsqueeze(0)], dim=0)
            # The Reward we got
            R = torch.cat([R, reward.unsqueeze(0)], dim=0)
            # The Value we thought it would be ???????????
            V = torch.cat([V, val.unsqueeze(0)], dim=0)

            # if old_state is None:
            #     old_state = mN
            # else:
            #     sars = [old_state, action.unsqueeze(-1), reward, mN]
            #     self.memory_replay_buffer.push(*sars)
            #     old_state = mN
        if not self._test:
            self.logger.add_log('tot_reward', self.problem.btwn_cent)
        # discount past rewards, rewards of the past are worth less
        for i in reversed(range(0, R.shape[0]-1)):
            R[i] = R[i] + self.gamma * R[i+1]
        return PI, R, V

    def predict_action(self, pi, illegal_actions):
        # th_layer = nn.Threshold(1, 0)
        # pi = th_layer(pi)
        # Remove the dimensions of size one
        pi = pi.squeeze()
        # For all the indices that are illegal set the distribution to negative infinity
        # No possible way they can be selected
        pi[illegal_actions] = -float('Inf')  # Whenever actor is trying to find policy distribution
        # Given this state, what action should i take-
        # if have illegal action(neighbor) don't want to take that action into account
        pi = F.softmax(pi, dim=0)  # Calculate distribution
        # pi[illegal_actions] = 0  # Whenever actor is trying to find policy distribution
        # Get the probability of action we can take
        dist = torch.distributions.categorical.Categorical(pi)
        # Take the higher probability
        probs = dist.probs.detach().numpy()
        # probs = probs / sum(probs)
        if self._test:
            action = probs.argmax()
        else:
            action = dist.sample()
        return action

    def update_model(self, PI, R, V):
        self.gcn_optimizer.zero_grad()  # needed to update parameter correctly
        if self.cuda:
            R = R.cuda()
        A = (R.squeeze() - V.squeeze()).detach()
        L_policy = -(torch.log(PI) * A).mean()
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI * PI.log()).mean()
        L = L_policy + L_value  # - 0.1 * L_entropy
        L.backward()
        self.gcn_optimizer.step()
        self.lr_schedule.step()
        self.logger.add_log('td_error', L_value.detach().item())
        self.logger.add_log('entropy', L_entropy.cpu().detach().item())

    # def memory_replay(self):
    #     self.optimizer.zero_grad()  # needed to update parameter correctly
    #     num_exp = 500
    #     if len(self.memory_replay_buffer) >= num_exp:
    #         experiences = sample(self.memory_replay_buffer, num_exp)
    #     else:
    #         experiences = self.memory_replay_buffer
    #     PI, R, V = list(map(lambda x: torch.cat(x), zip(*experiences)))
    #     PI, R, V = list(map(lambda x: x.unsqueeze(-1), [PI, R, V]))
    #     V = V.unsqueeze(-1)
    #
    #     A = (R.squeeze() - V.squeeze()).detach()
    #     L_policy = -(torch.log(PI) * A).mean()
    #     L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
    #     L = L_policy + L_value  # - 0.1 * L_entropy
    #     L.backward()
    #     self.optimizer.step()

    # Run so many numbers of episode then run the model
    def train(self):
        [PI, R, V] = self.run_episode()  # getting new json file, getting new graph/ subgraph
        for i in range(self.num_episodes - 1):  # for each range in episodes, why do have episodes = 1???
            [pi, r, v] = self.run_episode()
            # Update model
            PI = torch.cat([PI, pi], dim=0)  # Appending what learned to previous pi
            R = torch.cat([R, r], dim=0)
            V = torch.cat([V, v], dim=0)
        # self.memory_replay()
        self.update_model(PI, R, V)
        return self.logger

    def test(self):
        [_, _, _] = self.run_episode()
        return self.problem.btwn_cent

    def save_model(self):  # takes what we learned
        torch.save(self.model.state_dict(), self.path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
