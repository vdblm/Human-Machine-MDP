"""
Implementation of the known and unknown human experiments in the paper
"""
import os

from environments.env_types import EnvironmentType
from agents.switching_agents import Agent, Optimal, Alg2, Greedy
from agents.hum_mac_agent import NoisyDriverAgent, UniformDriverAgent
from environments.episodic_mdp import GridMDP
from environments.make_envs import make_cell_based_env, make_sensor_based_env, grid_feature_extractor, \
    make_feature_extractor
from plot.plot_path import PlotPath, HUMAN_COLOR, MACHINE_COLOR
import numpy as np


class SwitchingExperiment:
    """
    Known and unknown switching experiments.
    Child classes are `CellBasedSwitchingExperiment` and `SensorBasedSwitchingExperiment`
    """

    def __init__(self, env_type: EnvironmentType, machine_agent: NoisyDriverAgent, human_agent: NoisyDriverAgent):
        self.env_type = env_type
        self.agent0 = machine_agent
        self.agent1 = human_agent
        self.feat_ext = None

    def learn_evaluate(self, is_learn: bool, env: GridMDP, switching_agent: Agent, switching_cost: float,
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

    def run_known_human_exp(self, human_cost: float, switching_cost: float, n_try: int = 100, plot: bool = True,
                            plot_name: str = None):
        """
        Run the known human experiment in the paper.
        It generates a grid environment based on `self.env_type`
        and runs the optimal switching policy in it.

        Parameters
        ----------
        human_cost: float
        switching_cost: float
        n_try: int
            Number of repeating the experiment
        plot: bool
            If `plot==True`, then the trajectory induced by
            the optimal switching policy will be plotted.
        plot_name : str
            Name of the output plot. If `None`, then
            it will be "h_{human_cost}_s_{switching_cost}"

        Returns
        -------
        result : dict
            A dictionary that contains the expected total cost,
            expected environment cost, human control rate, and
            expected number of switching.
        """


class SensorBasedSwitchingExperiment(SwitchingExperiment):
    def __init__(self, env_type: EnvironmentType, machine_agent: NoisyDriverAgent, human_agent: NoisyDriverAgent):
        super().__init__(env_type, machine_agent, human_agent)
        self.feat_ext = make_feature_extractor(env_type=env_type)

        # for a sensor-based state space, switching policy is trained only over
        # the environment type not the grid environment
        self.train_env = make_sensor_based_env(self.env_type)

        # update agents' action policies
        self.agent0.update_policy(self.train_env, sensor_based=True)
        self.agent1.update_policy(self.train_env, sensor_based=True)

        # uniform agent
        self.uniform_agent = UniformDriverAgent(self.train_env, sensor_based=True)

    def run_known_human_exp(self, human_cost: float, switching_cost: float, n_try: int = 100, plot: bool = True,
                            plot_name: str = None):

        # generate an env with cell-based state space based on `self.env_type`
        env = make_cell_based_env(self.env_type)

        # find the optimal switching policy
        optimal_switching_agent = Optimal(env=self.train_env, agent0=self.agent0, agent1=self.agent1,
                                          switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost)
        optimal_switching_agent.update_policy()

        # evaluate and plot
        plt_path = PlotPath(env, n_try)
        result = self.learn_evaluate(is_learn=False, env=env, switching_agent=optimal_switching_agent,
                                     switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                     n_try=n_try, plt_path=plt_path)
        if plot:
            if plot_name is None:
                plot_name = 'h_{}_s_{}'.format(human_cost, switching_cost)
            plt_path.plot(plot_name)
        return result

    def run_unknown_human_exp(self, n_episode: int, switching_cost: float, human_cost: float, n_try: int,
                              plot_epochs: list = None, plot_dir: str = None, verbose: bool = True,
                              log_freq: int = 100):
        """
        Run the unknown human experiment in the paper. In each episode,
        it generates a new grid environment and runs the optimal algorithm (algorithm 1),
        algorithm 2, and greedy algorithm to find the switching policy. To train the
        switching agents, we only run once in an episode. However, the evaluation is done
        by repeating the experiment `n_try` times in each episode (We don't train during evaluation).

        Parameters
        ---------
        n_episode: int
        switching_cost: float
        human_cost: float
        n_try: int
            Number of repeating the experiment in each episode to evaluate the switching algorithms
        plot_epochs: list
            List of episode numbers in which the trajectory of algorithm 2 will be plotted.
        plot_dir: str
            Directory for saving the plots. If `None` no plot will be saved.
        verbose : bool
            If `True`, then it will print logs
        log_freq : int
            How many episodes between logging

        Returns
        -------
        alg2_regret : list
            A list containing the regret of algorithm 2 in each episode, i.e.,
            `alg2_regret[i]` = expected cost of algorithm 2 - optimal expected cost in episode 'i'
        greedy_regret : list
            A list containing the regret of the greedy algorithm in each episode
        """

        if plot_epochs is None:
            plot_epochs = []

        optimal_switching_agent = Optimal(env=self.train_env, agent0=self.agent0, agent1=self.agent1,
                                          switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost)
        optimal_switching_agent.update_policy()

        """
        Note that both Alg2 and Greedy switching agents don't have access to the true human policy. 
        Instead, they begin with a uniform action policy.
        """
        # the unknown agent is human (agent1)
        alg2_switching_agent = Alg2(env=self.train_env, agent0=self.agent0, agent1=self.uniform_agent,
                                    switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost, delta=0.1,
                                    unknown_agent=1)

        greedy_switching_agent = Greedy(env=self.train_env, agent0=self.agent0, agent1=self.uniform_agent,
                                        switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost, delta=0.1,
                                        unknown_agent=1)

        alg2_regret = []
        greedy_regret = []

        for ep in range(n_episode):
            alg2_switching_agent.update_policy(ep)
            greedy_switching_agent.update_policy(ep)

            # initialize a new grid env for this episode
            grid_env = make_cell_based_env(self.env_type)

            # evaluate the optimal policy
            optimal_result = self.learn_evaluate(is_learn=False, env=grid_env,
                                                 switching_agent=optimal_switching_agent,
                                                 switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                                 n_try=n_try)
            optimal_value = optimal_result['exp_cost']

            # plot the trajectory induced by algorithm 2
            if ep in plot_epochs:
                plt_path = PlotPath(grid_env, n_try)
            else:
                plt_path = None
            # evaluate algorithm2 and greedy algorithm
            alg2_result = self.learn_evaluate(is_learn=False, env=grid_env, switching_agent=alg2_switching_agent,
                                              switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                              n_try=n_try, plt_path=plt_path)

            greedy_result = self.learn_evaluate(is_learn=False, env=grid_env, switching_agent=greedy_switching_agent,
                                                switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                                n_try=n_try)

            alg2_regret.append(alg2_result['exp_cost'] - optimal_value)
            greedy_regret.append(greedy_result['exp_cost'] - optimal_value)

            if plt_path is not None:
                plot_name = 'ep_{}'.format(ep)
                plot_file = os.path.join(plot_dir, plot_name)
                plt_path.plot(plot_file)

            # learn algorithm2 and greedy algorithm
            self.learn_evaluate(is_learn=True, env=grid_env, switching_agent=alg2_switching_agent,
                                switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                n_try=1)
            self.learn_evaluate(is_learn=True, env=grid_env, switching_agent=greedy_switching_agent,
                                switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                n_try=1)
            if verbose and ep % log_freq == 0:
                print("################### Episode {} ###################".format(ep))
                print('algorithm 2 cumulative regret: {}'.format(np.sum(alg2_regret)))
                print('greedy algorithm cumulative regret: {}'.format(np.sum(greedy_regret)))
            grid_env.reset()

        return alg2_regret, greedy_regret


class CellBasedSwitchingExperiment(SwitchingExperiment):
    def __init__(self, env_type: EnvironmentType, machine_agent: NoisyDriverAgent, human_agent: NoisyDriverAgent):
        super().__init__(env_type, machine_agent, human_agent)
        self.feat_ext = grid_feature_extractor

    def run_known_human_exp(self, human_cost: float, switching_cost: float, n_try: int = 100, plot: bool = True,
                            plot_name: str = None):

        # generate an env with cell-based state space based on `self.env_type`
        env = make_cell_based_env(self.env_type)

        # update agents' action policies
        self.agent0.update_policy(env, sensor_based=False)
        self.agent1.update_policy(env, sensor_based=False)

        # find the optimal switching policy
        optimal_switching_agent = Optimal(env=env, agent0=self.agent0, agent1=self.agent1,
                                          switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost)
        optimal_switching_agent.update_policy()

        # evaluate and plot
        plt_path = PlotPath(env, n_try)
        result = self.learn_evaluate(is_learn=False, env=env, switching_agent=optimal_switching_agent,
                                     switching_cost=switching_cost, agent0_cost=0, agent1_cost=human_cost,
                                     n_try=n_try, plt_path=plt_path)
        if plot:
            if plot_name is None:
                plot_name = 'h_{}_s_{}'.format(human_cost, switching_cost)
            plt_path.plot(plot_name)
        return result
