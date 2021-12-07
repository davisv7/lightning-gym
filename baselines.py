import numpy as np


class EsroyAgent:
    def __init__(self, problem):
        self.problem = problem  # environment

    def run_episode(self):  # similar to epochs
        done = False
        _ = self.problem.reset()  # We get our initial state by resetting

        while not done:  # While we haven't exceeded budget
            # Get action from policy network
            action = self.pick_greedy_action()

            # take action
            _, reward, done, _ = self.problem.step(action.item())  # Take action and find outputs

        return self.problem.btwn_cent

    def pick_greedy_action(self):
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        best_action = None
        if self.problem.num_actions + self.problem.budget_offset < 2:
            pass
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
            _, reward, done, _ = self.problem.step(action.item())  # Take action and find outputs

        return self.problem.btwn_cent

    def pick_random_action(self):
        legal_actions = self.problem.get_legal_actions().squeeze().detach().numpy()
        return np.random.choice(legal_actions)
