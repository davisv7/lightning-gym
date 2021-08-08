# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: orrivlin
"""
import torch
import torch.nn.functional as F
from lightning_gym.GCN import GCN
from lightning_gym.ACN import ACN


class DiscreteActorCritic:
    def __init__(self, problem, cuda_flag=False, load_model=False, **kwargs):

        self.problem = problem  # environment
        self.cuda = cuda_flag
        self._load_model = load_model  # if have previous model to pass down
        self.path = 'mvc_net.pt'

        # hyperparameters
        self.in_feats = kwargs.get("ndim", 4)  # of node features - equal to length of x in BTWN.py
        self.n_hidden = kwargs.get("hdim", 256)
        self.gamma = kwargs.get("gamma", 0.99)
        '''
        gamma - defines how the contribution of past rewards are discounted if gamma is 1, then there is no discount
        '''
        self.learning_rate = kwargs.get("lr", 1.5e-3)  # this changes the learning rate
        self.num_episodes = 1  # is it redundant to have # of episodes, in main running episodes?
        self._test = kwargs.get("test", False)

        # create the model for the ajay
        self.model = GCN(self.in_feats, self.n_hidden, self.n_hidden, n_layers=4, activation=F.leaky_relu)
        # self.acnet = ACN(self.n_hidden)
        if self._load_model:  # making model
            self.load_model()
        if cuda_flag:
            self.model = self.model.cuda()

        # Does optimizer make it work better?
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.model.policy.parameters()},
        #     {'params': self.model.value.parameters()}
        # ], lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # logs information about trials/networks
        # self.log = logger()
        # self.log.add_log('tot_return')
        # self.log.add_log('TD_error')
        # self.log.add_log('entropy')
        # self.log.add_log('gains')

    def print_actor_configuration(self, ):

        print("\tLoad model: {}".format(self._load_model),
              "Learning Rate: {}".format(self.learning_rate),
              sep="\n\t"
              )

    def run_episode(self):  # similar to epochs
        done = False
        G = self.problem.reset()  # We get our initial state by resetting
        # Add our illegal actions which are ones in the edge vector
        [illegal_actions, _] = self.problem.get_illegal_actions()  # getting neighbors
        # Initialize the list below
        PI = torch.empty(0)  # policy network
        R = torch.empty(0)  # reward
        V = torch.empty(0)  # value network

        while not done:  # While we haven't exceeded budget
            # Use this if we have an NVIDIA graphics card (we don't)
            if self.cuda:
                G.ndata['features'] = G.ndata['features'].cuda()

            # convolve our model
            # G = self.model(G)

            # We put it into our model
            [pi, val] = self.model(G)

            # Get action from policy network
            action = self.predict_action(pi, illegal_actions)

            # take action
            G, reward, done, _ = self.problem.step(action.item())  # Take action and find outputs
            [illegal_actions, _] = self.problem.get_illegal_actions()

            # collect outputs of networks for learning - cat = appending for tensors
            # Probability for taking action
            PI = torch.cat([PI, pi[action].unsqueeze(0)], dim=0)
            # The Reward we got
            R = torch.cat([R, reward.unsqueeze(0)], dim=0)
            # The Value we thought it would be ???????????
            V = torch.cat([V, val.unsqueeze(0)], dim=0)

        tot_return = R.sum().item()
        self.problem.r_logger.add_log('tot_return', tot_return)
        # self.log.add_item('gains',np.flip(R.numpy()))

        # discount past rewards, rewards of the past are worth less
        for i in range(R.shape[0] - 1):
            R[-2 - i] = R[-2 - i] + self.gamma * R[-1 - i]

        return PI, R, V, tot_return

    def predict_action(self, pi, illegal_actions):
        # Remove the dimensions of size one
        pi = pi.squeeze()
        # For all the indices that are illegal set the distribution to negative infinity
        # No possible way they can be selected
        pi[illegal_actions] = -float('Inf')  # Whenever actor is trying to find policy distribution
        # Given this state, what action should i take-
        # if have illegal action(neighbor) don't want to take that action into account
        pi = F.softmax(pi, dim=0)  # Calculate distribution
        # Get the probability of action we can take
        dist = torch.distributions.categorical.Categorical(pi)
        # Take the higher probability
        if self._test:
            action = dist.probs.argmax()
        else:
            action = dist.sample()
        return action

    def update_model(self, PI, R, V):
        # R = (R - R.mean()) / (R.std() + 0.00001)
        self.optimizer.zero_grad()  # needed to update parameter correctly
        if self.cuda:
            R = R.cuda()
        A = R.squeeze() - V.squeeze().detach()
        L_policy = -(torch.log(PI) * A).mean()
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI * PI.log()).mean()
        L = L_policy + L_value  # - 0.1 * L_entropy
        L.backward()
        self.optimizer.step()
        self.problem.r_logger.add_log('td_error', L_value.detach().item())
        self.problem.r_logger.add_log('entropy', L_entropy.cpu().detach().item())

    # Run so many numbers of episode then run the model
    def train(self):
        [PI, R, V, _] = self.run_episode()  # getting new json file, getting new graph/ subgraph
        for i in range(self.num_episodes - 1):  # for each range in episodes, why do have episodes = 1???
            [pi, r, v, _] = self.run_episode()
            # Update model
            PI = torch.cat([PI, pi], dim=0)  # Appending what learned to previous pi
            R = torch.cat([R, r], dim=0)
            V = torch.cat([V, v], dim=0)

        self.update_model(PI, R, V)
        return self.problem.r_logger

    def test(self):
        [_, _, _, _] = self.run_episode()
        return self.problem.r_logger

    def save_model(self):  # takes what we learned
        torch.save(self.model.state_dict(), self.path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
