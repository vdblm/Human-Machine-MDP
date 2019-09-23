"""
an episodic RL MDP, including
states, actions, costs, transition probabilities, and episode length.

author: balazadehvahid@gmail.com
"""
import numpy as np
from matplotlib import collections as mc
import pylab as pl


class EpisodicMDP:
    def __init__(self, n_states, n_actions, ep_l, costs, transition_probs):
        """
        :param n_states: number of states
        :param n_actions: number of actions
        :param ep_l: episode length
        :type n_states: int
        :type n_actions: int
        :type ep_l: int
        :param costs: True costs
        :type costs: dict
        :param transition_probs: True transitions
        :type transition_probs: dict
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_l = ep_l

        # transition probs and costs
        self.__costs = costs
        self.__probs = transition_probs

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
        cost = self.__costs[self.state, action]
        self.state = np.random.choice(self.n_states, p=self.__probs[self.state, action])

        self.time_step += 1
        if self.time_step >= self.ep_l:
            finished = True
        else:
            finished = False

        return self.state, cost, finished


def make_chessboard(length=5, ep_l=15, eps=0.1, trans_prob=.8):
    """
    Chessboard example in the paper
    :param length: Number of cells = length * length
    :type length: int
    :param ep_l: episode length
    :type ep_l: int
    :param eps: the least cost
    :type eps: float
    :param trans_prob: The probability of a successful transition
    :type trans_prob: float
    """

    # actions = {up, down, left, right}
    n_actions = 4
    n_states = length * length

    # true costs and transitions
    cost_true = {}
    transition_true = {}

    def next_state(x, y, a):
        y_n = y + int(a < 2) * (-2 * a + 1)
        x_n = x + int(a >= 2) * (2 * a - 5)
        s_n = length * y_n + x_n
        # if the next cell is wall
        if x_n >= length or x_n < 0 or y_n >= length or y_n < 0:
            s_n = s
        return s_n

    # states = (xy) in base length. (00) is the start cell.
    for x in range(length):
        for y in range(length):
            c = eps
            if x > y and (x - y + 1) % 2 == 0:
                c += (x - y + 1) / 2 * eps
            elif y > x and (y - x) % 2 == 0:
                c += (y - x) / 2 * eps
            s = y * length + x
            for a in range(n_actions):
                cost_true[s, a] = c

            # transitions
            for a in range(n_actions):
                transition_true[s, a] = np.zeros(n_states, dtype=np.float)
                s_n = next_state(x, y, a)
                # transition in the goal state
                if s == length * length - 1:
                    transition_true[s, a][s_n] = 1
                else:
                    transition_true[s, a][s_n] = trans_prob
                    for a2 in range(n_actions):
                        if a2 != a:
                            s_n = next_state(x, y, a2)
                            transition_true[s, a][s_n] = (1 - trans_prob) / (n_actions - 1)
                # normalize probs
                transition_true[s, a] = np.divide(transition_true[s, a], sum(transition_true[s, a]))
    # goal cost
    for a in range(n_actions):
        cost_true[length * length - 1, a] = 0

    chessboard = EpisodicMDP(n_states, n_actions, ep_l, cost_true, transition_true)
    return chessboard, cost_true, transition_true


class PlotPath:
    """
    plot agent path in the chessboard experiment
    """

    def __init__(self, episodes, length=5):
        """

        :param length: number of cells = length * length
        :type length: int
        """
        self.length = length
        self.episode = episodes

        # lines {(org_long, org_lat, dest_long, dest_lat, color): count}
        self.lines = {}

    def add_line(self, org_state, dest_state, color):
        """
        adds line to the plot
        :type org_state: int
        :type dest_state: int
        """
        org_y = int(org_state / self.length)
        org_x = org_state % self.length
        dest_y = int(dest_state / self.length)
        dest_x = dest_state % self.length
        line_count = self.lines.get((org_x, org_y, dest_x, dest_y, color))
        if line_count is None:
            line_count = 1
        else:
            line_count += 1
        self.lines[(org_x, org_y, dest_x, dest_y, color)] = line_count

    def plot(self):
        lines = []
        colors = []
        width = []
        for line_tuple in self.lines:
            line = [(line_tuple[0] + 0.5, line_tuple[1] + 0.5), (line_tuple[2] + 0.5, line_tuple[3] + 0.5)]
            lines.append(line)
            colors.append(line_tuple[4])
            width.append(self.lines.get(line_tuple))
        width = np.divide(width, self.episode/5)
        lc = mc.LineCollection(lines, colors=colors, linewidths=width)
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        pl.grid(True)
        return pl, fig
