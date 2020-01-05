from agents.switching_agents import Agent
import numpy as np


class ChessboardHumanAgent(Agent):
    def __init__(self, env):
        """

        :param env: a chessboard.py environments
        :type env: environments.EpisodicMDP
        """
        super().__init__()
        self.env = env

        l = np.sqrt(self.env.n_states)
        for s in range(self.env.n_states):
            self.policy[s] = np.zeros(self.env.n_actions)
            if s >= l * (l - 1):
                # there's a wall on the upside, goes right (action = 3)
                self.policy[s][3] = 1
            else:
                # goes up (action = 0)
                self.policy[s][0] = 1

    def take_action(self, curr_state):
        return np.random.choice(range(self.env.n_actions), p=self.policy[curr_state])


class ChessboardMachineAgent(Agent):
    def __init__(self, env):
        """

        :param env: a chessboard.py environments
        :type env: environments.EpisodicMDP
        """
        super().__init__()
        self.env = env

        l = np.sqrt(self.env.n_states)
        for s in range(self.env.n_states):
            self.policy[s] = np.zeros(self.env.n_actions)
            if (s + 1) % l == 0:
                # there's a wall on the right side, goes up (action = 0)
                self.policy[s][0] = 1
            else:
                # goes right (action = 3)
                self.policy[s][3] = 1

    def take_action(self, curr_state):
        return np.random.choice(range(self.env.n_actions), p=self.policy[curr_state])
