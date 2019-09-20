"""
author: balazadehvahid@gmail.com
"""
import numpy as np


class Agent:
    """
    agent superclass
    """

    def __init__(self):
        self.policy = {}
        pass

    def update_obs(self, *args):
        """Add observations to the record"""
        pass

    def update_policy(self):
        """Update its action policy"""
        pass

    def take_action(self, *args):
        """Returns an action based on its policy"""
        pass


class Belief:
    """
    Belief superclass
    """

    def __init__(self):
        pass

    def update(self, *args):
        """update belief using observation"""
        pass

    def sample(self):
        """sample from the belief space"""
        pass


class FixedBelief(Belief):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def sample(self):
        return self.obj


class FiniteHorizonSwitchingAgent(Agent):
    """
    a switching agent, which decides the controller of the system.

    child agents will implement update_policy function.
    """

    def __init__(self, env, agent0_prior, agent1_prior, cost_prior, transition_prob_prior, switching_cost, human_cost):
        """
        :param env: The environment object
        :type env: environment.EpisodicMDP
        :param agent0_prior: The prior of the first agent, like machine.
        :type agent0_prior: Belief
        :param agent1_prior:The prior of the second agent, like human.
        :type agent1_prior: Belief
        :param cost_prior: The prior belief based on the costs.
        :type cost_prior: Belief
        :param transition_prob_prior: The prior belief based on the transitions.
        :type transition_prob_prior: Belief
        :param switching_cost: The cost of switching between two agent
        :type switching_cost: float
        :param human_cost: The cost of choosing the human as the controller
        :type human_cost: float
        """
        super().__init__()
        self.agents_prior = [agent0_prior, agent1_prior]
        self.cost_prior = cost_prior
        self.transition_prob_prior = transition_prob_prior
        self.env = env
        self.switching_cost = switching_cost
        self.human_cost = human_cost

        # initialize the current agent (0 or 1)
        self.cur_agent = 0

        # initialize the switching policy. dictionary {(time_step, prev_agent, state):agent1_prob}
        # for time_step = 0, prev_agent = -1!
        for t in range(env.ep_l):
            for state in range(env.n_states):
                if t == 0:
                    self.policy[t, state, -1] = 1
                else:
                    for prev_agent in [0, 1]:
                        self.policy[t, state, prev_agent] = 1

        # initialize the agent0, agent1, costs and transition probs
        self.agents = [self.agents_prior[0].sample(), self.agents_prior[1].sample()]
        self.costs = self.cost_prior.sample()
        self.transition_prob = self.transition_prob_prior.sample()

    def take_action(self, state, prev_agent, time_step):
        """chooses the controller of the current state"""
        agent0_prob = self.policy[time_step, state, prev_agent]
        self.cur_agent = np.random.choice([0, 1], p=[agent0_prob, 1 - agent0_prob])
        return self.cur_agent

    def update_obs(self, prev_state, prev_agent, action, cost, cur_state, finished):
        """ update the posterior beliefs"""

        # update agents_prior
        self.agents_prior[prev_agent].update(prev_state, action)

        # update cost_prior
        self.cost_prior.update(prev_state, action, cost)

        # update transition_prior
        if not finished:
            self.transition_prob_prior.update(prev_state, action, cur_state)


class ExactPSRL(FiniteHorizonSwitchingAgent):
    def update_policy(self):
        self.agents = [self.agents_prior[0].sample(), self.agents_prior[1].sample()]
        self.costs = self.cost_prior.sample()
        self.transition_prob = self.transition_prob_prior.sample()

        human = self.agents[1]
        machine = self.agents[0]

        # initialize value functions value[time_step, prev_agent, state]
        value = {}
        for t in range(self.env.ep_l):
            if t == 0:
                value[t] = np.zeros(self.env.n_states, dtype=np.float)
            else:
                for d in [0, 1]:
                    value[t, d] = np.zeros(self.env.n_states, dtype=np.float)

        # dynamic programming
        for t in range(self.env.ep_l - 1, -1, -1):
            for s in range(self.env.n_states):
                # d = 1 is for human and d = 0 for machine
                g = 0
                f = 0
                for a in range(self.env.n_actions):
                    if t < self.env.ep_l - 1:
                        g_exp = np.dot(value[t + 1, 0],
                                       self.transition_prob[s, a])
                        f_exp = np.dot(value[t + 1, 1],
                                       self.transition_prob[s, a])
                    else:
                        g_exp = 0
                        f_exp = 0
                    g += machine.policy[s][a] * (
                            self.costs[s, a] + g_exp)
                    f += human.policy[s][a] * (
                            self.costs[s, a] + f_exp)

                if t > 0:
                    for d in [0, 1]:
                        left = self.switching_cost * int(d == 1) + g
                        right = self.human_cost + self.switching_cost * int(d == 0) + f
                        if left <= right:
                            self.policy[t, s, d] = 1
                            value[t, d][s] = left
                        else:
                            self.policy[t, s, d] = 0
                            value[t, d][s] = right
                else:
                    left = g
                    right = self.human_cost + f
                    if left <= right:
                        self.policy[t, s, -1] = 1
                        value[t][s] = left
                    else:
                        self.policy[t, s, -1] = 0
                        value[t][s] = right


class ChessboardHumanAgent(Agent):
    def __init__(self, env):
        """

        :param env: a chessboard environment
        :type env: environment.EpisodicMDP
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

        :param env: a chessboard environment
        :type env: environment.EpisodicMDP
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
