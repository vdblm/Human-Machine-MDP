from environments.episode_mdp import EpisodicMDP
import numpy as np


def in_range(i, ranges):
    for range in ranges:
        if range[0] <= i <= range[1]:
            return True
    return False


def make_default_cell_types(width, height):
    cell_types = {}
    # cell_types default value
    if cell_types is None:
        cell_types = {}
    stone_width_ranges = [(0, np.math.ceil(width / 7) - 1), (np.math.ceil(6 * width / 7), width - 1)]
    grass_width_ranges = [(np.math.ceil(width / 7), np.math.ceil(2 * width / 7) - 1),
                          (np.math.ceil(5 * width / 7), np.math.ceil(6 * width / 7) - 1)]
    lane_width_ranges = [(np.math.floor((width - 1) / 2), np.math.floor((width + 1) / 2) - 1)]
    lane_height_range = [(np.math.ceil((height - 1) / 3), height - 1 - np.math.ceil((height - 1) / 3))]

    for x in range(width):
        for y in range(height):
            if in_range(x, stone_width_ranges):
                cell_type = 'stone'
            elif in_range(x, grass_width_ranges):
                cell_type = 'grass'
            elif not in_range(x, lane_width_ranges) and in_range(y, lane_height_range):
                cell_type = 'grass'
            else:
                cell_type = 'road'
            cell_types[x, y] = cell_type
            # end cell
            cell_types[np.math.floor((width - 1) / 2), height - 1] = 'end'

    return cell_types


def make_lane_keeping(width=7, height=10, costs=None, cell_types=None):
    """
    lane keeping example in the paper. each cell is a 'road', 'grass', 'stone', or 'end'.

    :type width: int
    :type height: int
    :param cell_types: {(x, y): type} defines type ('road', etc) of each state (x, y)
    :type cell_types: dict
    :param costs: {type: cost}
    :type costs: dict

    episode length is `height`.

    """

    # TODO: width should be at least 7

    # actions = [left turn, keep straight, right turn]
    n_actions = 3
    n_states = width * height
    ep_l = height

    if costs is None:
        costs = {'road': 0.1, 'grass': 0.2, 'stone': 0.3, 'end': 0}

    if cell_types is None:
        cell_types = make_default_cell_types(width, height)

    # true costs and transitions
    cost_true = {}
    transition_true = {}

    def next_state(x, y, a):
        y_n = y + 1
        x_n = x + a - 1
        # if the next cell is wall
        if x_n >= width or x_n < 0:
            x_n = x
        if y_n >= height:
            y_n = y
        return x_n, y_n

    # states = (x, y) will be y * width + x
    for x in range(width):
        for y in range(height):
            c = costs[cell_types.get((x, y))]
            s = y * width + x
            for a in range(n_actions):
                cost_true[s, a] = c

            # transitions
            for a in range(n_actions):
                transition_true[s, a] = np.zeros(n_states, dtype=np.float)
                x_n, y_n = next_state(x, y, a)

                s_n = y_n * width + x_n
                transition_true[s, a][s_n] = 1

    lane_keeping = EpisodicMDP(n_states, n_actions, ep_l, cost_true, transition_true)
    return lane_keeping, cost_true, transition_true
