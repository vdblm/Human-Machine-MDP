"""
Implementation of the human/machine policies in the paper
"""
import random

from agents.switching_agents import Agent
import numpy as np

from environments.episodic_mdp import EpisodicMDP, GridMDP
from environments.make_envs import FeatureStateHandler, TYPE_COSTS


class NoisyDriverAgent(Agent):
    def __init__(self, noise_sd: float = 0, car_fear: float = None):
        """
        A noisy driver, which chooses the cell with the lowest noisy estimated cost.

        Parameters
        ----------
        noise_sd : float
            Standard deviation of the Gaussian noise (i.e., `N(0, noise_sd)`)
        car_fear : float
            Probability of moving to a 'car' type cell (i.e., $p_{panic}$ in the paper)
        """
        super().__init__()
        self.car_fear = car_fear
        self.noise_sd = noise_sd
        # [left, straight, right]
        self.n_action = 3

    def update_policy(self, env: EpisodicMDP, sensor_based, type_costs=TYPE_COSTS, env_types=None):

        if not sensor_based:
            assert isinstance(env, GridMDP), 'invalid environment class'
        self.policy = np.zeros((env.n_state, self.n_action))
        for s in range(env.n_state):
            if not sensor_based:
                type_costs[None] = np.nan
                x = s % env.width
                y = s // env.width
                # straight action on the top lane
                if y >= env.height - 1:
                    self.policy[s][1] = 1
                    continue
                types = [env.cell_types.get((x + a - 1, y + 1)) for a in range(self.n_action)]
            else:
                type_costs['wall'] = np.nan

                # TODO better handle this
                if env_types is None:
                    feature_hdnlr = FeatureStateHandler()
                else:
                    feature_hdnlr = FeatureStateHandler(env_types)
                f_s = feature_hdnlr.state2feature(s)
                types = [f_s[i] for i in range(1, 4)]

            noisy_actions = []
            feared = False
            for a in range(self.n_action):
                if types[a] == 'car' and self.car_fear is not None:
                    self.policy[s][a] = self.car_fear
                    feared = True
                else:
                    noisy_actions.append(a)

            next_cells = [type_costs[t] for t in types]
            flag = False
            for a in noisy_actions:
                if not np.isnan(next_cells[a]):
                    flag = True
            if len(noisy_actions) > 0 and flag:
                if self.noise_sd != 0:
                    # TODO may sample more than 100!
                    tmp = np.asarray(
                        [np.nanargmin(
                            [next_cells[a] + np.random.normal(0, self.noise_sd) for a in noisy_actions])
                            for j in range(100)])
                    for i, a in enumerate(noisy_actions):
                        self.policy[s][a] = len(np.where(tmp == i)[0]) / len(tmp)
                        if feared:
                            self.policy[s][a] *= (1 - self.car_fear)
                else:
                    next_cells = [next_cells[a] for a in noisy_actions]
                    min_as = np.where(next_cells == np.nanmin(next_cells))[0]
                    for i in min_as:
                        self.policy[s][noisy_actions[i]] = 1 / len(min_as)
                        if feared:
                            self.policy[s][noisy_actions[i]] *= (1 - self.noise_sd)

            if np.sum(self.policy[s]) == 0:
                self.policy[s] = np.asarray([1 / 3, 1 / 3, 1 / 3])
            self.policy[s] /= np.sum(self.policy[s])

    def take_action(self, curr_state):
        return random.choices(range(self.n_action), self.policy[curr_state])[0]


class UniformDriverAgent(Agent):
    def __init__(self, env: EpisodicMDP, sensor_based, type_costs=TYPE_COSTS, env_types=None):
        """
        A uniform driver, which chooses the next cell uniformly at random.
        """
        super().__init__()
        if not sensor_based:
            assert isinstance(env, GridMDP), 'invalid environment class'

        self.n_action = 3
        self.policy = np.zeros((env.n_state, self.n_action))
        for s in range(env.n_state):
            if not sensor_based:
                type_costs[None] = np.nan
                x = s % env.width
                y = s // env.width
                # straight action on the top lane
                if y >= env.height - 1:
                    self.policy[s][1] = 1
                    continue
                types = [env.cell_types.get((x + a - 1, y + 1)) for a in range(self.n_action)]
            else:
                type_costs['wall'] = np.nan
                # TODO better handle this
                if env_types is None:
                    feature_hdnlr = FeatureStateHandler()
                else:
                    feature_hdnlr = FeatureStateHandler(env_types)
                f_s = feature_hdnlr.state2feature(s)
                types = [f_s[i] for i in range(1, 4)]

            next_cells = [type_costs[t] for t in types]
            non_nan_num = np.count_nonzero(~np.isnan(next_cells))
            if non_nan_num > 0:
                for a in range(self.n_action):
                    if ~np.isnan(next_cells)[a]:
                        self.policy[s][a] = 1 / non_nan_num
            else:
                self.policy[s] = np.asarray([1 / 3, 1 / 3, 1 / 3])
            self.policy[s] /= np.sum(self.policy[s])

    def take_action(self, curr_state):
        return random.choices(range(self.n_action), self.policy[curr_state])[0]
