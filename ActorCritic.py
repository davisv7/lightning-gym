# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: orrivlin
"""
import torch
import torch.nn.functional as F
from copy import deepcopy as dc
from lightning_gym.GCN import GCN
from lightning_gym.Logger import Logger
import numpy as np


class DiscreteActorCritic:
    def __init__(self, problem, cuda_flag=False, load_model=False, **kwargs):

        self.problem = problem
        self.cuda = cuda_flag
        self._load_model = load_model
        self.path = 'mvc_net.pt'

        # hyperparameters
        self.in_feats = kwargs.get("ndim", 3)  # of node features - equal to length of x in BTWN.py
        self.n_hidden = kwargs.get("hdim", 256)
        self.gamma = kwargs.get("gamma", 1)
        self.learning_rate = kwargs.get("lr", 0.001)
        self.num_episodes = 1
        self._test = kwargs.get("test", False)

        # create the model for the agent
        self.model = GCN(self.in_feats, self.n_hidden, self.n_hidden, n_layers=3, activation=F.rrelu)
        if self._load_model:
            self.load_model()
        if cuda_flag:
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # logs information about trials/networks
        # self.log = logger()
        # self.log.add_log('tot_return')
        # self.log.add_log('TD_error')
        # self.log.add_log('entropy')
        # self.log.add_log('gains')

    def run_episode(self):
        done = False
        state = self.problem.reset() #We get our initial state by reseting
        #Add our illegal acitons which are ones in the edge vector
        [illegal_actions, _] = self.problem.get_illegal_actions()
        #Initialize the list below
        PI = torch.empty(0)
        R = torch.empty(0)
        V = torch.empty(0)

        while not done:
            #Pull the graph
            G =state

            #If we have graphic card
            if self.cuda:
                G.ndata['features'] = G.ndata['features'].cuda()

            #We put it into our model
            [pi, val] = self.model(G)

            # get action from policy network

            #Remove the dimmensions of size one
            pi = pi.squeeze()
            #For all the indices that are illegal set the distribution to neg inf.
            #No possible way they can be selected
            pi[illegal_actions] = -float('Inf')
            pi = F.softmax(pi, dim=0)
            #Get the probabilty of action we can take
            dist = torch.distributions.categorical.Categorical(pi)
            #Take the higher probabilty
            if self._test:
                action = dist.probs.argmax()
            else:
                action = dist.sample()

            # take action
            new_state, reward, done, _ = self.problem.step(action.item())
            [illegal_actions, _] = self.problem.get_illegal_actions()
            state = new_state

            # collect outputs of networks for learning - cat-> appending
            # Probability for taking action
            PI = torch.cat([PI, pi[action].unsqueeze(0)], dim=0)
            #The Reward we got
            R = torch.cat([R, reward.unsqueeze(0)], dim=0)
            #The Value we thought it would be
            V = torch.cat([V, val.unsqueeze(0)], dim=0)
            # A = torch.cat([A, action.unsqueeze(0)], dim=0)

        tot_return = R.sum().item()
        # self.log.add_item('tot_return', tot_return)
        # self.log.add_item('gains',np.flip(R.numpy()))

        # discount past rewards
        # Actions taken in the past has less to do in our actions in the future
        for i in range(R.shape[0] - 1):
            R[-2 - i] = R[-2 - i] + self.gamma * R[-1 - i]

        return PI, R, V, tot_return

    def update_model(self, PI, R, V):
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        A = R.squeeze() - V.squeeze().detach()
        L_policy = -(torch.log(PI) * A).mean()
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI * PI.log()).mean()
        L = L_policy + L_value - 0.1 * L_entropy
        L.backward()
        self.optimizer.step()
        # self.log.add_item('TD_error', L_value.detach().item())
        # self.log.add_item('entropy', L_entropy.cpu().detach().item())

    #Run so many numbers of episode then run the model
    def train(self):
        [PI, R, V, _] = self.run_episode()
        for i in range(self.num_episodes - 1):
            [pi, r, v, _] = self.run_episode()
            PI = torch.cat([PI, pi], dim=0)
            R = torch.cat([R, r], dim=0)
            V = torch.cat([V, v], dim=0)

        self.update_model(PI, R, V)
        # return self.log

    def test(self):
        [_, _, _, _] = self.run_episode()
        # return self.log

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
