import pickle

import numpy as np

from agents.hum_mac_agent import FeatureAgent, ObstacleAvoidAgent
from agents.switching_agents import Optimal
from environments.make_envs import make_sensor_based_env, FeatureStateHandler, make_cell_based_env
from known_human_experiment import LearnSwitching
from obstacle_avoid_task import heavy_traffic_obstacle_env, traffic_free_env, heavy_traffic_env
import time

hum_env = []

# choose env type
horizon = 9
types = ['road', 'grass', 'stone']
featurizer = FeatureStateHandler(types=types)
# type_probs = {'road': 0.5, 'grass': 0.3, 'car': 0.2}
side_probs = {'road': 0.4, 'grass': 0.3, 'stone': 0.3}
middle_probs = {'road': 1, 'grass': 0, 'stone': 0}
costs = {'road': 0, 'grass': 2, 'stone': 4, 'car': 5}
featured_mdp, featured_cost, featured_transitions = make_sensor_based_env(ep_l=horizon,
                                                                          type_costs=costs,
                                                                          type_probabilities=side_probs,
                                                                          middle_probs=middle_probs,
                                                                          featurizer=featurizer)
width, height, start_point = 3, horizon + 1, (1, 0)
n_try = 100
# define agents
machine_feature_agent = FeatureAgent(featured_mdp.n_states, costs, featurizer, noise_sd=4, car_fear=None)
human_feature_agent = FeatureAgent(featured_mdp.n_states, costs, featurizer, noise_sd=1, car_fear=0.4)

# switch costs # human costs
switch_costs = [0]
human_costs = [0]
variances = np.arange(0, 7, 0.5)
for var in variances:
    for human_cost in human_costs:
        for switch_cost in switch_costs:
            """
            Run the experiment
            agent0 is the default agent
            agent1 is unknown
            """

            # TODO a cleaner code!
            human_feature_agent = FeatureAgent(featured_mdp.n_states, costs, featurizer, noise_sd=var, car_fear=0.4)
            local_optimal_agent = Optimal(featured_mdp, machine_feature_agent, human_feature_agent, featured_cost,
                                          featured_transitions,
                                          switch_cost, 0, human_cost)
            local_optimal_agent.update_policy()
            print(str(time.time()) + '###########' + str(var))
            for i in range(1000):
                # Define a sample cell types based on the probabilities
                cell_types = traffic_free_env()  #
                env, cost_true, transition_true = make_cell_based_env(width, height, costs, cell_types, start_point)
                # machine_grid_agent = ObstacleAvoidAgent(width, height, costs, cell_types, noise_sd=4)
                # human_grid_agent = ObstacleAvoidAgent(width, height, costs, cell_types, noise_sd=1, car_fear=0.4)
                #
                # global_optimal_agent = Optimal(env, machine_grid_agent, human_grid_agent, cost_true, transition_true,
                #                                switch_cost, 0, human_cost)
                # global_optimal_agent.update_policy()
                #
                # # TODO feature to learn_switch
                # learn_switch = LearnSwitching(env, machine_grid_agent, human_grid_agent, switch_cost, human_cost, width,
                #                               height, cell_types)
                # env_cost, human_cntrl, switch_n = learn_switch.evaluate(global_optimal_agent, n_try, env_cost=True)
                # hum_env['grid'].add((human_cntrl, env_cost))

                learn_switch = LearnSwitching(env, machine_feature_agent, human_feature_agent, switch_cost, human_cost,
                                              width,
                                              height, cell_types, featurizer=featurizer)
                env_cost, human_cntrl, switch_n = learn_switch.evaluate(local_optimal_agent, n_try, env_cost=True)
                hum_env.append((switch_n, human_cntrl, env_cost, var))

pickle.dump(hum_env, open('outputs/env1_var_env_00', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
