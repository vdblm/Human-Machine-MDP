from agents.switching_agents import Agent
import numpy as np


class LaneKeepingAgent(Agent):
    def __init__(self, width, height, cell_types, prob_weights):
        """

        :param cell_types: {(x, y): cell type}
        :type cell_types: dict
        :param prob_weights: {cell type: prob weight} like {'stone':1, 'road':2, 'grass':3}
        :type prob_weights: dict
        """
        super().__init__()
        prob_weights[None] = 0

        self.n_states = width * height
        self.n_actions = 3

        for s in range(self.n_states):
            self.policy[s] = np.zeros(self.n_actions)
            x = s % width
            y = int(s / width)
            # TODO straight action at last row ?
            if y >= height - 1:
                # straight action
                self.policy[s][1] = 1
                continue
            weights = []
            for i in range(0, 3):
                if i is 0:
                    cond = x > 0
                elif i is 2:
                    cond = x < width - 1
                else:
                    cond = True
                tmp = cell_types[x + i - 1, y + 1] if cond else None
                weights.append(prob_weights[tmp])
            self.policy[s] = weights / np.sum(weights)

    def take_action(self, curr_state):
        return np.random.choice(range(self.n_actions), p=self.policy[curr_state])
