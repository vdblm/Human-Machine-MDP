"""
Environment types in the paper for both cell-based and sensor-based state spaces.
"""

from environments.episodic_mdp import GridMDP, EpisodicMDP
from environments.env_types import EnvironmentType
import numpy as np

# default number of features in sensor-based state space
N_FEATURE = 4
# default cell types
CELL_TYPES = ['road', 'grass', 'stone', 'car']
# default width and height
WIDTH, HEIGHT = 3, 10
# default costs
TYPE_COSTS = {'road': 0, 'grass': 2, 'stone': 4, 'car': 5}
# default episode length
EP_L = HEIGHT - 1


class FeatureStateHandler:
    """
    Convert a feature vector to a state number and vice versa.
    For example, if the cell types are ['road', 'grass', 'car'] and feature vector dim = 4, then
    s = ('road', 'grass', 'road', 'car') <--> s = (0102) in base 3 = 11
    """

    def __init__(self, types: list = CELL_TYPES, n_feature: int = N_FEATURE):
        """
        Parameters
        ----------
        types : list of str
            All the cell types
        n_feature : int
            Dimension of the feature vector
        """

        # types= ['road', 'car', ...]
        if 'wall' not in types:
            types.append('wall')
        self.types = types

        self.type_value = {}
        for i, t in enumerate(self.types):
            self.type_value[t] = i

        self.base = len(self.types)
        self.n_feature = n_feature

        self.n_state = np.power(self.base, self.n_feature)

    def feature2state(self, feature_vec: list):
        """
        Parameters
        ----------
        feature_vec : list of str
            An input feature vector. The dimension should be
            equal to `self.n_feature`

        Returns
        -------
        state_num : int
            The state number corresponding to the input feature vector
        """

        assert len(feature_vec) == self.n_feature, 'input dimension not equal to the feature dimension'

        values = [self.type_value[g] for g in feature_vec]
        state_num = 0
        for i, value in enumerate(values):
            state_num += value * np.power(self.base, len(values) - i - 1)
        return state_num

    def state2feature(self, state: int):
        """
        Parameters
        ----------
        state : int
            An state number. It should satisfy the condition
            `0 <= state < self.n_state`

        Returns
        -------
        feature_vec : list
            The feature vector corresponding to the input state number
        """
        assert 0 <= state < self.n_state, 'invalid state number'

        type_numbers = list(map(int, np.base_repr(state, self.base)))
        for j in range(self.n_feature - len(type_numbers)):
            type_numbers.insert(0, 0)

        feature_vec = [self.types[i] for i in type_numbers]
        return feature_vec


def make_feature_extractor(env_type: EnvironmentType):
    types = list(env_type.type_probs.keys())

    def sensor_feature_extractor(env, state):
        width = env.width
        cell_types = env.cell_types

        x = state % width
        y = state // width
        f_s = [cell_types[x, y]]
        for i in range(3):
            f_s.append(cell_types.get((x + i - 1, y + 1), 'wall'))
        return FeatureStateHandler(types=types).feature2state(f_s)

    return sensor_feature_extractor


def grid_feature_extractor(env, state):
    return state


def make_cell_based_env(env_type: EnvironmentType, width: int = WIDTH, height: int = HEIGHT,
                        type_costs: dict = TYPE_COSTS, start_cell: tuple = None):
    """
    Generate an episodic MDP with cell-based state space.
    Episode length is `height - 1`.

    Parameters
    ----------
    type_costs : dict[str, float]
        Cost of each cell type. For example: {'road': 0, 'grass': 2, ...}
    env_type : EnvironmentType
        Specifies the environment type defined in the paper

    start_cell : tuple
        (x, y) coordinate of the starting cell. If `start_cell=None`,
        then it will be the middle cell of the first row.

    Returns
    -------
    env : GridMDP
        The generated cell-based episodic MDP based on the environment type
    """
    # actions = [left, straight, right]
    n_action = 3
    n_state = width * height
    ep_l = height - 1

    if start_cell is None:
        start_cell = (np.floor((width - 1) / 2), 0)

    # true costs and transitions
    true_costs = {}
    true_transitions = {}

    # generate cell types
    cell_types = env_type.generate_cell_types(width, height)

    # the states corresponding to cell (x, y) will be `y * width + x`
    for x in range(width):
        for y in range(height):
            state = y * width + x

            # costs
            cost = type_costs[cell_types[x, y]]
            for action in range(n_action):
                true_costs[state, action] = cost

            # transitions
            for action in range(n_action):
                true_transitions[state, action] = np.zeros(n_state, dtype=np.float)

                # the new cell after taking 'action'
                x_n, y_n = x + action - 1, y + 1

                # the top row
                if y_n >= height:
                    y_n = y

                # handle wall
                is_wall = x_n < 0 or x_n >= width
                if is_wall:
                    if x_n < 0:
                        s_n1 = y_n * width + 0
                        s_n2 = y_n * width + 1
                    elif x_n >= width:
                        s_n1 = y_n * width + width - 1
                        s_n2 = y_n * width + width - 2
                    true_transitions[state, action][s_n1] = 0.5
                    true_transitions[state, action][s_n2] = 0.5
                else:
                    s_n = y_n * width + x_n
                    true_transitions[state, action][s_n] = 1

    start_state = start_cell[0] + start_cell[1] * width

    env = GridMDP(n_state, n_action, ep_l, costs=true_costs, trans_probs=true_transitions,
                  start_state=start_state, width=width, height=height, cell_types=cell_types)
    return env


def make_sensor_based_env(env_type: EnvironmentType, ep_l: int = EP_L, type_costs: dict = TYPE_COSTS,
                          n_feature: int = N_FEATURE):
    """
    Generate an episodic MDP with sensor-based state space.

    Parameters
    ----------
    ep_l : int
        Episode length
    type_costs : dict[str, float]
        Cost of each cell type. For example: {'road': 0, 'grass': 2, ...}
    env_type : EnvironmentType
        Specifies the environment type defined in the paper
    n_feature : int
        Dimension of the sensor-based measurements (i.e., feature vector)

    Returns
    -------
    env : EpisodicMDP
        The sensor-based episodic MDP based on the environment type
    """

    def __find_pos(f):
        # TODO only works for width=3. FIX IT
        # positions: 0=left, 1=middle, 2=right
        if f[1] == 'wall':
            return 0
        if f[3] == 'wall':
            return 2
        return 1

    def __calc_prob(f_s, f_sn, a):
        # raise exception when the action goes to wall
        pos = __find_pos(f_s)
        pos_n = __find_pos(f_sn)
        if a != 1 and pos == a:
            raise Exception('The action goes to wall')

        check_correct_pos = pos + a - 1 == pos_n
        if not check_correct_pos or f_sn[0] != f_s[a + 1]:
            return 0

        middle_cell = 3 - pos_n
        middle_prob = type_probs[f_sn[middle_cell]] if middle_probs is None else middle_probs[f_sn[middle_cell]]
        # calculate the probability
        if pos_n == 0:
            return 1 * type_probs[f_sn[2]] * middle_prob
        if pos_n == 1:
            return type_probs[f_sn[1]] * middle_prob * type_probs[f_sn[3]]

        return middle_prob * type_probs[f_sn[2]] * 1

    # actions = [left, straight, right]
    n_action = 3

    type_probs = env_type.type_probs
    middle_probs = env_type.mid_lane_type_probs

    # add 'wall' type
    type_probs['wall'] = 0
    if middle_probs is not None:
        middle_probs['wall'] = 0
    type_costs['wall'] = max(type_costs.values()) + 1

    f_s_handler = FeatureStateHandler(types=list(type_probs.keys()), n_feature=n_feature)

    # number of states
    n_state = f_s_handler.n_state

    # true costs and transitions
    true_transitions = {}
    true_costs = {}

    for state in range(n_state):
        for action in range(n_action):
            feat_vec = f_s_handler.state2feature(state)

            true_costs[state, action] = type_costs[feat_vec[0]]
            true_transitions[state, action] = np.zeros(n_state, dtype=np.float)

            for nxt_state in range(n_state):
                nxt_feat_vec = f_s_handler.state2feature(nxt_state)

                # handle wall
                # TODO the code works only when `n_feature = 4`. FIX IT
                if __find_pos(feat_vec) == 0 and action == 0:
                    a1, a2 = 1, 2
                elif __find_pos(feat_vec) == 2 and action == 2:
                    a1, a2 = 0, 1
                else:
                    a1, a2 = action, action

                true_transitions[state, action][nxt_state] = 0.5 * (__calc_prob(feat_vec, nxt_feat_vec, a1) +
                                                                    __calc_prob(feat_vec, nxt_feat_vec, a2))

            # Normalize
            true_transitions[state, action] = true_transitions[state, action] / sum(true_transitions[state, action])

    env = EpisodicMDP(n_state, n_action, ep_l, true_costs, true_transitions, start_state=0)
    return env
