"""
Implementation of the switching agents in the paper
"""
import math
import random
import numpy as np
import copy
from environments.episodic_mdp import EpisodicMDP
from numba import njit


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
        self.env = env
        self.switching_cost = switching_cost
        self.agent0_cost = agent0_cost
        self.agent1_cost = agent1_cost

        # initialize the current agent as agent0
        self.cur_agent = 0

        # initialize the switching policy [time_step][prev_agent][state]: agent0_prob
        self.policy = np.ones(shape=(self.env.ep_l, 2, self.env.n_state))

    def take_action(self, state, prev_agent, time_step):
        """chooses the controller of the current state"""
        agent0_prob = self.policy[time_step][prev_agent][state]
        self.cur_agent = random.choices([0, 1], [agent0_prob, 1 - agent0_prob])[0]
        return self.cur_agent

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        """ update agents histories """
        pass


class Alg2(SwitchingAgent):
    def __init__(self, env, agent0, agent1, switching_cost, agent0_cost, agent1_cost, delta, unknown_agent=0,
                 scale=0.1):
        super().__init__(env, agent0, agent1, switching_cost, agent0_cost, agent1_cost)

        self.unknown_agent = unknown_agent
        self.scale = scale
        self.delta = delta
        self.episode_num = 0

        self.unknown_prior = np.zeros(shape=(self.env.n_state, self.env.n_action))

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        if d_t != self.unknown_agent:
            return
        self.unknown_prior[s_t][action] += 1

    def compute_prior(self):
        arr = np.copy(self.unknown_prior)
        arr[arr.sum(axis=1) == 0] = 1
        return arr / arr.sum(axis=1)[0:, np.newaxis]

    def compute_beta(self):
        ret = self.unknown_prior.sum(axis=1)
        ret[ret == 0] = 1
        t_k = self.episode_num * self.env.ep_l + 1
        return np.sqrt(self.scale * self.env.n_action * math.log(self.env.n_state * t_k / self.delta) / ret)

    @staticmethod
    @njit
    def extend_val_itr(P_unk, P_known, beta, e_c, e_p, n_s, n_ac, e_l, n_ag, u_ag, s_c, ag0_c, ag1_c):
        # q_val[time][prev_agent][state][curr_agent]
        q_val = np.zeros(shape=(e_l, n_ag, n_s, n_ag))
        # q_min[time][prev_agent][state]
        q_min = np.zeros(shape=(e_l + 1, n_ag, n_s))

        # policy[time][prev_agent][state]
        policy = np.ones(shape=(e_l, n_ag, n_s))
        for i in range(e_l):
            t = e_l - i - 1
            for s in range(n_s):
                for d in range(n_ag):
                    action_vals = np.array([e_c[s][a] + np.dot(e_p[s][a], q_min[t + 1][d]) for a in range(n_ac)])
                    if d != u_ag:
                        p_opt = P_known[s]
                    else:
                        sorted_actions = np.argsort(action_vals)

                        # optimize P
                        p_opt = np.copy(P_unk[s])
                        if p_opt[sorted_actions[0]] + 0.5 * beta[s] > 1:
                            p_opt = np.zeros(n_ac)
                            p_opt[sorted_actions[0]] = 1
                        else:
                            p_opt[sorted_actions[0]] += 0.5 * beta[s]

                        i = n_ac - 1
                        while p_opt.sum() > 1:
                            p_opt[sorted_actions[i]] = max(1 - p_opt.sum() + p_opt[sorted_actions[i]], 0)
                            i -= 1

                    value = np.dot(action_vals, p_opt)
                    for dm in range(n_ag):
                        meta_cost = d * ag1_c + (1 - d) * ag0_c + int(dm != d) * s_c * int(t != 0)
                        q_val[t][dm][s][d] = meta_cost + value
                for dm in range(n_ag):
                    # update policy
                    best_action = np.argmin(q_val[t][dm][s])
                    policy[t][dm][s] = 1 - best_action
                    q_min[t][dm][s] = q_val[t][dm][s][best_action]

        return policy

    def update_policy(self, ep_num):
        self.episode_num = ep_num
        beta = self.compute_beta()
        u_ag = self.unknown_agent
        P_unk = self.compute_prior()
        P_known = self.agents[1 - u_ag].policy

        e_c, e_p = self.env.costs, self.env.trans_probs
        n_s, n_ac, e_l, n_ag = self.env.n_state, self.env.n_action, self.env.ep_l, 2
        s_c, ag0_c, ag1_c = self.switching_cost, self.agent0_cost, self.agent1_cost
        self.policy = self.extend_val_itr(P_unk, P_known, beta, e_c, e_p, n_s, n_ac, e_l, n_ag, u_ag, s_c, ag0_c, ag1_c)


class Greedy(Alg2):
    def compute_beta(self):
        return np.zeros(self.env.n_state)


class Optimal(Alg2):
    def __init__(self, env, agent0, agent1, switching_cost, agent0_cost, agent1_cost):
        super().__init__(env, agent0, agent1, switching_cost, agent0_cost, agent1_cost, delta=0, unknown_agent=0)

    def compute_beta(self):
        return np.zeros(self.env.n_state)

    def compute_prior(self):
        return self.agents[self.unknown_agent].policy


# TODO add doc, add cost
class UCRL2(SwitchingAgent):
    def __init__(self, env, agent0, agent1, switching_cost, agent0_cost, agent1_cost, delta, scale,
                 unknown_agent=0):
        super().__init__(env, agent0, agent1, switching_cost, agent0_cost, agent1_cost)

        self.delta = delta
        self.episode_num = 0
        self.unknown_agent = unknown_agent
        self.scale = scale

        # transition prior for the unknown agent, P[curr_state][nxt_state]
        self.P_unk_prior = np.zeros(shape=(env.n_state, env.n_state))

        # known transitions
        self.P_known = np.array([self.agents[1 - self.unknown_agent].policy[s].dot(self.env.trans_probs[s])
                                 for s in range(self.env.n_state)])

    def update_obs(self, d_tminus1, s_t, d_t, action, env_cost, s_tplus1, finished):
        if d_t != self.unknown_agent:
            return
        self.P_unk_prior[s_t][s_tplus1] += 1

    def compute_prior(self):
        arr = np.copy(self.P_unk_prior)
        arr[arr.sum(axis=1) == 0] = 1
        return arr / arr.sum(axis=1)[0:, np.newaxis]

    def compute_beta(self):
        ret = self.P_unk_prior.sum(axis=1)
        ret[ret == 0] = 1
        t_k = self.episode_num * self.env.ep_l + 1
        return np.sqrt(self.scale * self.env.n_state * math.log(self.env.n_state * t_k / self.delta) / ret)

    @staticmethod
    @njit
    def extend_val_itr(P_unk, P_known, beta, e_c, n_s, e_l, n_ag, u_ag, s_c, ag0_c, ag1_c):

        # value iteration
        q_values = np.zeros(shape=(e_l, n_ag, n_s, n_ag))
        q_min = np.zeros(shape=(e_l + 1, n_ag, n_s))
        policy = np.ones(shape=(e_l, n_ag, n_s))
        for i in range(e_l):
            t = e_l - 1 - i
            for dm in range(n_ag):
                sorted_ind = [np.argsort(q_min[t + 1][d]) for d in range(n_ag)]
                for s in range(n_s):
                    for d in range(n_ag):
                        # TODO here c only depends on the state not the action. FIX IT
                        env_C = e_c[s][0]
                        tot_cost = env_C + d * ag1_c + (1 - d) * ag0_c + abs(dm - d) * s_c * int(t != 0)
                        if d != u_ag:
                            P_opt = P_known[s]
                        else:
                            P_opt = np.copy(P_unk[s])
                            if P_opt[sorted_ind[d][0]] + 0.5 * beta[s] > 1:
                                P_opt = np.zeros(n_s)
                                P_opt[sorted_ind[d][0]] = 1
                            else:
                                P_opt[sorted_ind[d][0]] += 0.5 * beta[s]

                            # TODO handle P
                            ind = n_s - 1
                            summ = np.sum(P_opt)
                            while summ > 1:
                                diff = summ - P_opt[sorted_ind[d][ind]]
                                P_opt[sorted_ind[d][ind]] = 1 - diff if diff < 1 else 0
                                summ = diff + P_opt[sorted_ind[d][ind]]
                                ind -= 1
                        q_values[t][dm][s][d] = tot_cost + np.dot(q_min[t + 1][d], P_opt)
                    best_action = np.argmin(q_values[t][dm][s])
                    policy[t][dm][s] = 1 - best_action
                    q_min[t][dm][s] = np.min(q_values[t][dm][s])
        return policy

    def update_policy(self, ep_num):
        self.episode_num = ep_num
        beta = self.compute_beta()
        u_ag = self.unknown_agent
        P_unk = self.compute_prior()
        P_known = self.P_known

        e_c = self.env.costs
        n_s, e_l, n_ag = self.env.n_state, self.env.ep_l, 2
        s_c, ag0_c, ag1_c = self.switching_cost, self.agent0_cost, self.agent1_cost
        self.policy = self.extend_val_itr(P_unk, P_known, beta, e_c, n_s, e_l, n_ag, u_ag, s_c, ag0_c, ag1_c)


class FixedAgent(Agent):
    def __init__(self, fixed_action):
        super().__init__()
        self.fixed_action = fixed_action

    def take_action(self, *args):
        return self.fixed_action

    def update_obs(self, *args):
        pass

    def update_policy(self, *args):
        pass
