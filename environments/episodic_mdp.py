"""
An episodic RL MDP, including states, actions, costs, transition probabilities, and episode length.
"""
import numpy as np


class EpisodicMDP:
    def __init__(self, n_state, n_action, ep_l, costs: dict, trans_probs: dict, start_state=0):
        """
        Parameters
        ----------
        n_state : int
            Number of states
        n_action : int
            Number of actions
        ep_l : int
            Episode length
        costs : dict
            State-action costs, i.e., {(state, action): costs}

        trans_probs : dict
            Transition probabilities, i.e., {(state, action): list of next state probabilities}

        start_state : int
        """
        self.n_state = n_state
        self.n_action = n_action
        self.ep_l = ep_l

        # transition probs and costs
        self.costs = costs
        self.trans_probs = trans_probs

        # current state and time
        self.start_state = start_state
        self.state = start_state
        self.time_step = 0

    def reset(self):
        """ resets the env """
        self.time_step = 0
        self.state = self.start_state

    def step_forward(self, action):
        """
        The env moves one step forward

        Returns
        -------
        next_state : int
            The new state of the MDP
        costs : int or float
            The induced costs of the taken action
        finished : bool
            If time exceeds the episode length
        """
        cost = self.costs[self.state, action]
        self.state = np.random.choice(self.n_state, p=self.trans_probs[self.state, action])

        self.time_step += 1
        if self.time_step >= self.ep_l:
            finished = True
        else:
            finished = False

        return self.state, cost, finished


class GridMDP(EpisodicMDP):
    """
    Grid episodic MDP with height, width, and cell types
    """

    def __init__(self, n_state, n_action, ep_l, costs: dict, trans_probs: dict,
                 start_state, width, height, cell_types: dict):
        """

        Parameters
        ----------
        height : int
            Height of the grid env
        width : int
            Width of the grid env
        cell_types : dict
            Type of each grid cell with coordinate (x, y), i.e., {(x, y): type}.
        """
        super().__init__(n_state, n_action, ep_l, costs, trans_probs, start_state)
        self.height = height
        self.width = width
        self.cell_types = cell_types
