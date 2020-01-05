from environments.episode_mdp import EpisodicMDP
import numpy as np


def make_chessboard(length=5, ep_l=15, eps=0.1, trans_prob=.8):
    """
    Chessboard example
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
