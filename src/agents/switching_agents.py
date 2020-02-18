"""
Implementation of the switching agents in the paper
"""
from collections import defaultdict
import numpy as np
import copy
from environments.episodic_mdp import EpisodicMDP


class Agent:
    """
    Agent superclass
    """

    def __init__(self):
        self.policy = {}

    def update_obs(self, *args):
        """Add observations to the record"""

    def update_policy(self, *args):
        """Update the action policy"""

    def take_action(self, *args):
        """Return an action based on the policy"""


class SwitchingAgent(Agent):
    """
    An agent that decides the control switch at each time, given the current state and
    the previous control switch (i.e., $\pi_t(d_t|s_t, d_{t-})$ in the paper).

    Child agents will implement `update_policy` function.
    """

    def __init__(self, env: EpisodicMDP, agent0: Agent, agent1: Agent,
                 switching_cost: float, agent0_cost: float, agent1_cost: float):
        """
        The initial controller agent is agent0

        Parameters
        ----------
        env : EpisodicMDP
            The environment object
        agent0 : Agent
            The first agent action policy (i.e., human/machine action policy)
        agent1 : Agent
            The second agent action policy (i.e., human/machine action policy)
        switching_cost: float
            The costs of switching between two agent
        agent0_cost: float
            The costs of choosing agent0 as the controller
        agent1_cost: float
            The costs of choosing agent1 as the controller
        """
        super().__init__()
        self.agents = [copy.deepcopy(agent0), copy.deepcopy(agent1)]
        self.costs = env.costs
        self.trans_probs = env.trans_probs
        self.env = env
        self.switching_cost = switching_cost
        self.agent0_cost = agent0_cost
        self.agent1_cost = agent1_cost

        # {0:[(s_t, a_t), ...], 1: ...}
        self.agent_history = {0: [], 1: []}

        # initialize the current agent as agent0
        self.cur_agent = 0

        # initialize the switching policy as a dictionary: {(time_step, state, prev_agent): agent0_prob}
        # if time_step is 0, prev_agent will be -1!
        for t in range(env.ep_l):
            for state in range(env.n_state):
                if t == 0:
                    self.policy[t, state, -1] = 1
                else:
                    for prev_agent in [0, 1]:
                        self.policy[t, state, prev_agent] = 1

    def take_action(self, state, prev_agent, time_step):
        """chooses the controller of the current state"""
        agent0_prob = self.policy[time_step, state, prev_agent]
        self.cur_agent = np.random.choice([0, 1], p=[agent0_prob, 1 - agent0_prob])
        return self.cur_agent

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        """ update agents histories """

        # update agents history
        self.agent_history[d_t].append((s_t, action))

        # TODO may update cost and transition history


# TODO refactor!
class Alg2(SwitchingAgent):
    def __init__(self, env, agent0, agent1, switching_cost, agent0_cost, agent1_cost, delta, unknown_agent=0):
        super().__init__(env, agent0, agent1, switching_cost, agent0_cost, agent1_cost)
        self.unknown_agent = unknown_agent
        self.value = {}
        # initialize value functions: value[time_step, prev_agent, state]
        for t in range(self.env.ep_l):
            if t == 0:
                self.value[t] = np.zeros(self.env.n_state, dtype=np.float)
            else:
                for d in [0, 1]:
                    self.value[t, d] = np.zeros(self.env.n_state, dtype=np.float)

        self.delta = delta
        self.episode_num = 0
        self.N_sa = np.zeros(shape=(env.n_state, env.n_action))
        self.emp_policy = np.true_divide(np.ones(shape=(self.env.n_state, self.env.n_action)), self.env.n_action)
        self.beta_t = np.zeros(self.env.n_state)

    def update_empirical(self):
        history = self.agent_history[self.unknown_agent]
        self.agent_history[1 - self.unknown_agent] = []
        while len(history) > 0:
            s, a = history.pop()
            self.N_sa[s][a] += 1

        # update empirical
        for s in range(self.env.n_state):
            N_s = np.sum(self.N_sa[s])
            if N_s > 0:
                self.emp_policy[s] = np.true_divide(self.N_sa[s], N_s)

            # compute beta
            t_k = self.episode_num * self.env.ep_l
            if t_k == 0:
                self.beta_t[s] = 0
            else:
                # TODO beta may be smaller than what in the paper
                coef = 1 / 10
                self.beta_t[s] = np.sqrt(
                    coef * self.env.n_action * np.log(self.env.n_state * t_k / self.delta) / max(N_s, 1))

    def find_agent(self, action_values, state):
        unknown_agent = self.agents[self.unknown_agent]
        unknown_agent.policy[state] = np.zeros(self.env.n_action)
        action_value = action_values[self.unknown_agent]

        sorted_actions = np.argsort(action_value)
        first_action = sorted_actions[0]
        unknown_agent.policy[state][first_action] = np.min([1,
                                                            np.add(self.emp_policy[state][first_action], np.true_divide(
                                                                self.beta_t[state], 2))])
        sum = unknown_agent.policy[state][first_action]
        i = 1
        while np.less(sum, 1) and not np.math.isclose(sum, 1, rel_tol=1e-6):
            unknown_agent.policy[state][sorted_actions[i]] = self.emp_policy[state][sorted_actions[i]]
            sum = np.add(sum, self.emp_policy[state][sorted_actions[i]])
            i += 1

        if sum > 1:
            unknown_agent.policy[state][sorted_actions[i - 1]] -= (sum - 1)
        unknown_agent.policy[state] /= np.sum(unknown_agent.policy[state])

        self.agents[self.unknown_agent] = unknown_agent

    def update_policy(self, ep_num):
        self.episode_num = ep_num
        self.update_empirical()
        # dynamic programming
        for t in range(self.env.ep_l - 1, -1, -1):
            for s in range(self.env.n_state):
                # d = 1 is for agent1 and d = 0 for agent0
                v0 = np.zeros(self.env.n_action, dtype=np.float)
                v1 = np.zeros(self.env.n_action, dtype=np.float)

                for a in range(self.env.n_action):
                    if t < self.env.ep_l - 1:
                        v0[a] = self.costs[s, a] + np.dot(self.value[t + 1, 0],
                                                          self.trans_probs[s, a])
                        v1[a] = self.costs[s, a] + np.dot(self.value[t + 1, 1],
                                                          self.trans_probs[s, a])
                    else:
                        v0[a] = self.costs[s, a]
                        v1[a] = self.costs[s, a]

                # Finds optimistic agents
                self.find_agent(action_values=[v0, v1], state=s)

                v0_exp = np.dot(self.agents[0].policy[s], v0)
                v1_exp = np.dot(self.agents[1].policy[s], v1)

                if t > 0:
                    for d in [0, 1]:
                        agnt0 = self.agent0_cost + self.switching_cost * int(d == 1) + v0_exp
                        agnt1 = self.agent1_cost + self.switching_cost * int(d == 0) + v1_exp
                        if agnt0 <= agnt1:
                            self.policy[t, s, d] = 1
                            self.value[t, d][s] = agnt0
                        else:
                            self.policy[t, s, d] = 0
                            self.value[t, d][s] = agnt1
                else:
                    agnt0 = self.agent0_cost + v0_exp
                    agnt1 = self.agent1_cost + v1_exp
                    if agnt0 <= agnt1:
                        self.policy[t, s, -1] = 1
                        self.value[t][s] = agnt0
                    else:
                        self.policy[t, s, -1] = 0
                        self.value[t][s] = agnt1


class Greedy(Alg2):
    def find_agent(self, action_values, state):
        self.agents[self.unknown_agent].policy[state] = self.emp_policy[state]


class Optimal(SwitchingAgent):
    def __init__(self, env, agent0, agent1, switching_cost, agent0_cost, agent1_cost):

        super().__init__(env, agent0, agent1, switching_cost, agent0_cost, agent1_cost)
        self.value = {}
        # initialize value functions: value[time_step, prev_agent, state]
        for t in range(self.env.ep_l):
            if t == 0:
                self.value[t] = np.zeros(self.env.n_state, dtype=np.float)
            else:
                for d in [0, 1]:
                    self.value[t, d] = np.zeros(self.env.n_state, dtype=np.float)

    def update_policy(self):
        # dynamic programming
        for t in range(self.env.ep_l - 1, -1, -1):
            for s in range(self.env.n_state):
                # d = 1 is for agent1 and d = 0 for agent0
                v0 = np.zeros(self.env.n_action, dtype=np.float)
                v1 = np.zeros(self.env.n_action, dtype=np.float)

                for a in range(self.env.n_action):
                    if t < self.env.ep_l - 1:
                        v0[a] = self.costs[s, a] + np.dot(self.value[t + 1, 0],
                                                          self.trans_probs[s, a])
                        v1[a] = self.costs[s, a] + np.dot(self.value[t + 1, 1],
                                                          self.trans_probs[s, a])
                    else:
                        v0[a] = self.costs[s, a]
                        v1[a] = self.costs[s, a]

                v0_exp = np.dot(self.agents[0].policy[s], v0)
                v1_exp = np.dot(self.agents[1].policy[s], v1)

                if t > 0:
                    for d in [0, 1]:
                        agnt0 = self.agent0_cost + self.switching_cost * int(d == 1) + v0_exp
                        agnt1 = self.agent1_cost + self.switching_cost * int(d == 0) + v1_exp
                        if agnt0 <= agnt1:
                            self.policy[t, s, d] = 1
                            self.value[t, d][s] = agnt0
                        else:
                            self.policy[t, s, d] = 0
                            self.value[t, d][s] = agnt1
                else:
                    agnt0 = self.agent0_cost + v0_exp
                    agnt1 = self.agent1_cost + v1_exp
                    if agnt0 <= agnt1:
                        self.policy[t, s, -1] = 1
                        self.value[t][s] = agnt0
                    else:
                        self.policy[t, s, -1] = 0
                        self.value[t][s] = agnt1
