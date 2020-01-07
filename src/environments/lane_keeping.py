from environments.episode_mdp import EpisodicMDP
import numpy as np


def __in_range(i, ranges):
    for range in ranges:
        if range[0] <= i <= range[1]:
            return True
    return False


def make_default_lane_keeping_cell_types(width, height):
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
            if __in_range(x, stone_width_ranges):
                cell_type = 'stone'
            elif __in_range(x, grass_width_ranges):
                cell_type = 'grass'
            elif not __in_range(x, lane_width_ranges) and __in_range(y, lane_height_range):
                cell_type = 'grass'
            else:
                cell_type = 'road'
            cell_types[x, y] = cell_type
            # end cell TODO we may not need an end cell
            # cell_types[np.math.floor((width - 1) / 2), height - 1] = 'end'

    return cell_types


def make_lane_keeping(width, height, costs, cell_types, start_cell=None):
    """
    lane keeping example in the paper. each cell is a 'road', 'grass', 'stone', or 'end'.
    TODO it's a deterministic env now, we may add some noise to transition probs
    :type width: int
    :type height: int
    :param cell_types: {(x, y): type} defines type ('road', etc) of each state (x, y)
    :type cell_types: dict
    :param costs: cost of each cell type {type: cost}
    :type costs: dict
    :param start_cell: (x, y) coordinate of the starting cell
    :type start_cell: tuple

    episode length is `height` - 1.

    """

    # TODO: width should be at least 7

    # actions = [left turn, keep straight, right turn]
    n_actions = 3
    n_states = width * height
    ep_l = height - 1

    if start_cell is None:
        start_cell = (np.math.ceil(2 * width / 7), 0)

    # true costs and transitions
    cost_true = {}
    transition_true = {}

    def __next_state(xx, yy, aa):
        yy_n = yy + 1
        xx_n = xx + aa - 1
        # if the next cell is wall
        if xx_n >= width or xx_n < 0:
            xx_n = xx
        if yy_n >= height:
            yy_n = yy
        return xx_n, yy_n

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
                x_n, y_n = __next_state(x, y, a)

                s_n = y_n * width + x_n
                transition_true[s, a][s_n] = 1

    lane_keeping = EpisodicMDP(n_states, n_actions, ep_l, cost_true, transition_true,
                               start_state=start_cell[0] + start_cell[1] * width)
    return lane_keeping, cost_true, transition_true
