"""
an episodic RL MDP, including
states, actions, costs, transition probabilities, and episode length.

author: balazadehvahid@gmail.com
"""
import numpy as np


class EpisodicMDP:
    def __init__(self, n_states, n_actions, ep_l):
        """
        :param n_states: number of states
        :param n_actions: number of actions
        :param ep_l: episode length
        :type n_states: int
        :type n_actions: int
        :type ep_l: int
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_l = ep_l

        # transition probs and costs
        self.__costs = {}
        self.__probs = {}

        # current state and time
        self.state = 0
        self.time_step = 0

    def reset(self):
        """
        resets the env.
        """
        self.time_step = 0
        self.state = 0

    def step_forward(self, action):
        """
        the env moves one step forward.
        :param action: taken action
        :type action: int
        :return new_state, cost, finished
        :rtype (int, float, bool)
        """
        cost = self.__costs[(self.state, action)]
        self.state = np.random.choice(self.n_states, p=self.__probs[(self.state, action)])

        self.time_step += 1
        if self.time_step >= self.ep_l:
            finished = True
        else:
            finished = False

        return self.state, cost, finished
