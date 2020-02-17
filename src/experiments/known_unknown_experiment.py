"""
Implementation of the known and unknown human experiments in the paper
"""

from environments.env_types import EnvironmentType
from agents.switching_agents import Agent, Optimal, UCB, Greedy
from agents.hum_mac_agent import NoisyDriverAgent
from environments.episodic_mdp import GridMDP
from environments.make_envs import make_cell_based_env, make_sensor_based_env
from plot.plot_path import PlotPath, HUMAN_COLOR, MACHINE_COLOR


class SwitchingExperiment:
    def __init__(self, env_type: EnvironmentType, machine_agent: NoisyDriverAgent, human_agent: NoisyDriverAgent,
                 feature_ext: function, sensor_based: bool = False):
        """
        Parameters
        ----------
        feature_ext : function
            A feature extractor function. It can be sensor-based or grid-based.
            A grid-based feature extractor returns the state itself, while
            a sensor-based return a 4-dim feature vector.
        sensor_based : bool
            Indicates if the state space is sensor-based.
        """
        self.env_type = env_type
        self.agent0 = machine_agent
        self.agent1 = human_agent
        self.feat_ext = feature_ext
        self.sensor_based = sensor_based

    def run_known_human_exp(self, human_cost: float, switching_cost: float, n_try: int = 100, plot: bool = True,
                            plot_file: str = None):
        """ run the known human experiment in the paper """

        # generate an env with cell-based state space based on `self.env_type`
        env = make_cell_based_env(self.env_type)

        # for a sensor-based state space, switching policy is trained only over
        # the environment type not the grid environment
        if self.sensor_based:
            train_env = make_sensor_based_env(self.env_type)
        else:
            train_env = env

        # update agents' action policies
        self.agent0.update_policy(train_env, sensor_based=self.sensor_based)
        self.agent1.update_policy(train_env, sensor_based=self.sensor_based)

        # find the optimal switching policy
        optimal_switching_agent = Optimal(env=train_env, agent0=self.agent0, agent1=self.agent1,
                                          switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost)
        optimal_switching_agent.update_policy()

        # evaluate and plot
        plt_path = PlotPath(env, n_try)
        result = self.__learn_evaluate(is_learn=False, env=env, switching_agent=optimal_switching_agent,
                                       switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                       n_try=n_try,
                                       plt_path=plt_path)
        if plot:
            if plot_file is None:
                plot_file = 'h_' + str(human_cost) + '_s_' + str(switching_cost)
            plt_path.plot(plot_file)
        return result

    def run_unknown_human_exp(self, n_episode, switching_cost, human_cost, n_try, plot_epochs, plot_path):
        # only for sensor-based
        assert self.sensor_based, 'consider a sensor-based experiment for the unknown case'
        train_env = make_sensor_based_env(self.env_type)

        # update agents' action policies
        self.agent0.update_policy(train_env, sensor_based=self.sensor_based)
        self.agent1.update_policy(train_env, sensor_based=self.sensor_based)

        optimal_switching_agent = Optimal(env=train_env, agent0=self.agent0, agent1=self.agent1,
                                          switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost)
        optimal_switching_agent.update_policy()

        # unknown agent is human (agent1)
        ucb_agent = UCB(env=train_env, agent0=self.agent0, agent1=self.agent1, switching_cost=switching_cost,
                        agent0_cost=0, agent1_cost=human_cost, delta=0.1, unknown_agent=1)

        greedy_agent = Greedy(env=train_env, agent0=self.agent0, agent1=self.agent1, switching_cost=switching_cost,
                              agent0_cost=0, agent1_cost=human_cost, delta=0.1, unknown_agent=1)

        ucb_regret = []
        greedy_regret = []

        for ep in range(n_episode):
            ucb_agent.update_episode_num(ep)
            ucb_agent.update_policy()
            greedy_agent.update_policy()

            # initialize a new grid env for this episode
            grid_env = make_cell_based_env(self.env_type)
            result = self.__learn_evaluate(is_learn=False, env=grid_env, switching_agent=optimal_switching_agent,
                                           switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                           n_try=n_try)
            optimal_value = result['exp_cost']

            if ep in plot_epochs:
                plt_path = PlotPath(grid_env, n_try)
            else:
                plt_path = None
            ucb_result = self.__learn_evaluate(is_learn=False, env=grid_env, switching_agent=ucb_agent,
                                               switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                               n_try=n_try)
            greedy_result = self.__learn_evaluate(is_learn=False, env=grid_env, switching_agent=greedy_agent,
                                                  switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                                  n_try=n_try)

            ucb_regret.append(ucb_result['exp_cost'] - optimal_value)
            greedy_regret.append(greedy_result['exp_cost'] - optimal_value)
            if plt_path is not None:
                plot_file = plot_path + '_ep_' + str(ep) + 'png'
                plt_path.plot(plot_file)

            self.__learn_evaluate(is_learn=True, env=grid_env, switching_agent=ucb_agent,
                                  switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                  n_try=1)
            self.__learn_evaluate(is_learn=True, env=grid_env, switching_agent=greedy_agent,
                                  switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                  n_try=1)
            grid_env.reset()

        return ucb_regret, greedy_regret

    def __learn_evaluate(self, is_learn: bool, env: GridMDP, switching_agent: Agent, switching_cost: float,
                         agent0_cost: float, agent1_cost: float, n_try=1, plt_path=None):
        """
        Learn or evaluate a switching policy in a grid environment.

        Parameters
        ----------
        is_learn : bool
            Indicates if we are learning or evaluating. If `is_learn == True`,
            then `n_try = 1`, and we will update the observations of `switching_agent`

        Returns
        -------
        result : dict
            A dictionary that contains the expected total cost,
            expected environment cost, human control rate, and
            expected number of switching.
        """

        if is_learn:
            n_try = 1

        total_costs = 0
        env_costs = 0
        switch_number = 0
        human_control = 0
        count = 0
        for i in range(n_try):
            finished = False
            env.reset()
            d_t = -1
            s_tplus1 = env.start_state
            while not finished:
                count += 1
                d_tminus1 = d_t
                s_t = s_tplus1
                d_t = switching_agent.take_action(self.feat_ext(env, s_t), d_tminus1, env.time_step)
                action = None
                if d_t == 0:
                    action = self.agent0.take_action(self.feat_ext(env, s_t))
                elif d_t == 1:
                    action = self.agent1.take_action(self.feat_ext(env, s_t))
                    human_control += 1

                s_tplus1, env_cost, finished = env.step_forward(action)
                is_switch = 1 if d_t != d_tminus1 and d_tminus1 != -1 else 0
                switch_number += is_switch
                cost = env_cost + switching_cost * is_switch + agent1_cost * d_t + agent0_cost * (1 - d_t)
                total_costs += cost
                env_costs += env_cost

                # update observations
                if is_learn:
                    switching_agent.update_obs(d_tminus1, self.feat_ext(env, s_t), d_t, action, env_cost,
                                               self.feat_ext(env, s_tplus1), finished)

                if plt_path is not None:
                    clr = MACHINE_COLOR if d_t == 0 else HUMAN_COLOR
                    plt_path.add_line(s_t, s_tplus1, clr)

        result = {'exp_cost': total_costs / n_try,
                  'env_cost': env_costs / n_try,
                  'human_control_rate': human_control / count,
                  'exp_switch': switch_number / n_try}

        return result
