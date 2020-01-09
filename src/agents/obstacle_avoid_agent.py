from agents.switching_agents import Agent
import numpy as np


class ObstacleAvoidAgent(Agent):
    def __init__(self, width, height, costs, cell_types, error_prob):
        """
        :param error_prob: probability of detecting front cells as roads
        :type error_prob: float
        """
        super().__init__()

        self.n_states = width * height
        self.n_actions = 3

        costs[None] = None

        def __next_state(x, y, a):
            y_n = y + 1
            x_n = x + a - 1
            # if the next cell is wall
            if x_n >= width or x_n < 0:
                return None
            if y_n >= height:
                return None
            return x_n, y_n

        for s in range(self.n_states):
            self.policy[s] = np.zeros(self.n_actions)
            x = s % width
            y = int(s / width)

            # TODO straight action at last row ?
            if y >= height - 1:
                # straight action
                self.policy[s][1] = 1
                continue

            observed_states_costs = {}
            for i in range(x - 2, x + 3):
                c = costs.get(cell_types.get((i, y + 2)))
                if c is None:
                    observed_states_costs[i, y + 2] = np.nan
                else:
                    observed_states_costs[i, y + 2] = np.random.choice([c / 2, 0], size=1,
                                                                       p=[1 - error_prob, error_prob])
            action_costs = []
            for a in range(self.n_actions):
                next_cell = __next_state(x, y, a)
                if next_cell is None:
                    action_costs.append(np.nan)
                    continue
                x_n, y_n = next_cell
                self_cost = costs.get(cell_types.get((x_n, y_n)))
                other_costs = [observed_states_costs.get((x_n + i, y_n + 1)) for i in range(-1, 2)]
                avg_other_costs = np.nanmean(other_costs)
                if np.isnan(avg_other_costs):
                    avg_other_costs = 0
                final_cost = self_cost + avg_other_costs
                action_costs.append(final_cost)

            min_actions = np.where(action_costs == np.nanmin(action_costs))[0]

            for a in min_actions:
                self.policy[s][a] = 1 / len(min_actions)

    def take_action(self, curr_state):
        return np.random.choice(range(self.n_actions), p=self.policy[curr_state])
