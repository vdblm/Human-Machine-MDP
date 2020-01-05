"""
author: balazadehvahid@gmail.com
"""
import numpy as np

from environments.episode_mdp import EpisodicMDP
from agents.beliefs import Belief


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


class FiniteHorizonSwitchingAgent(Agent):
    """
    a switching agent, which decides the controller of the system.

    child agents will implement update_policy function.
    """

    def __init__(self, env, agent0_prior, agent1_prior, cost_prior, transition_prob_prior, switching_cost, agent0_cost,
                 agent1_cost):
        """
        The initial agent is agent0

        :param env: The environments object
        :type env: EpisodicMDP
        :param agent0_prior: The prior knowledge of the first agent's action policy, like machine.
        :type agent0_prior: Belief
        :param agent1_prior:The prior knowledge of the second agent's action policy, like human.
        :type agent1_prior: Belief
        :param cost_prior: The prior belief based of the costs.
        :type cost_prior: Belief
        :param transition_prob_prior: The prior belief based of the transitions.
        :type transition_prob_prior: Belief
        :param switching_cost: The cost of switching between two agent
        :type switching_cost: float
        :param agent0_cost: The cost of choosing agent1 as the controller
        :type agent0_cost: float
        :param agent1_cost: The cost of choosing agent1 as the controller
        :type agent1_cost: float
        """
        super().__init__()
        self.agents_prior = [agent0_prior, agent1_prior]
        self.cost_prior = cost_prior
        self.transition_prob_prior = transition_prob_prior
        self.env = env
        self.switching_cost = switching_cost
        self.agent0_cost = agent0_cost
        self.agent1_cost = agent1_cost

        # initialize the current agent as agent0
        self.cur_agent = 0

        # initialize the switching policy. dictionary {(time_step, state, prev_agent):agent0_prob}
        # if time_step is 0, prev_agent will be -1!
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


class UCB(FiniteHorizonSwitchingAgent):
    def update_policy(self):
        # self.agents = [self.agents_prior[0].sample(), self.agents_prior[1].sample()]
        self.costs = self.cost_prior.sample()
        self.transition_prob = self.transition_prob_prior.sample()

        # human = self.agents[1]
        # machine = self.agents[0]

        # initialize value functions: value[time_step, prev_agent, state]
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
                # d = 1 is for agent1 and d = 0 for agent0
                v0 = np.zeros(self.env.n_actions, dtype=np.float)
                v1 = np.zeros(self.env.n_actions, dtype=np.float)

                for a in range(self.env.n_actions):
                    if t < self.env.ep_l - 1:
                        v0[a] = self.costs[s, a] + np.dot(value[t + 1, 0],
                                                          self.transition_prob[s, a])
                        v1[a] = self.costs[s, a] + np.dot(value[t + 1, 1],
                                                          self.transition_prob[s, a])
                    else:
                        v0[a] = self.costs[s, a]
                        v1[a] = self.costs[s, a]

                    # Finds optimistic agents TODO this UCB belief should be implemented in beliefs.py.
                    # It works for fixed belief
                    self.agents = [self.agents_prior[0].sample(v0, s, t), self.agents_prior[1].sample(v1, s, t)]

                v0_exp = np.dot(self.agents[0].policy[s], v0)
                v1_exp = np.dot(self.agents[1].policy[s], v1)

                if t > 0:
                    for d in [0, 1]:
                        agnt0 = self.agent0_cost + self.switching_cost * int(d == 1) + v0_exp
                        agnt1 = self.agent1_cost + self.switching_cost * int(d == 0) + v1_exp
                        if agnt0 <= agnt1:
                            self.policy[t, s, d] = 1
                            value[t, d][s] = agnt0
                        else:
                            self.policy[t, s, d] = 0
                            value[t, d][s] = agnt1
                else:
                    agnt0 = self.agent0_cost + v0_exp
                    agnt1 = self.agent1_cost + v1_exp
                    if agnt0 <= agnt1:
                        self.policy[t, s, -1] = 1
                        value[t][s] = agnt0
                    else:
                        self.policy[t, s, -1] = 0
                        value[t][s] = agnt1
