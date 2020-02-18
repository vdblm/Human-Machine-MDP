""" Running all the experiments in the paper"""
import matplotlib as mpl

mpl.use('pdf')
from experiments.change_sd import change_sd
from agents.hum_mac_agent import NoisyDriverAgent
from environments.env_types import ENV1, ENV2, ENV3, ENV3_LIGHT_TRAFFIC
from experiments.known_unknown_experiment import CellBasedSwitchingExperiment, SensorBasedSwitchingExperiment
from plot.plot_regret import plot_regret, plot_change_sd
import numpy as np

if __name__ == '__main__':
    machine = NoisyDriverAgent(noise_sd=4)
    human = NoisyDriverAgent(noise_sd=1, car_fear=0.4)

    # """known human cell-based state space (Figure 2.a in the paper)"""
    # cell_based_experiment = CellBasedSwitchingExperiment(env_type=ENV2, machine_agent=machine, human_agent=human)
    #
    # cell_based_experiment.run_known_human_exp(human_cost=0.2, switching_cost=0.2, n_try=100,
    #                                           plot_name='Env2_known_cell_based_0_2_0_2')
    #
    # """known human sensor-based state space (Figure 2.b in the paper)"""
    # sensor_based_experiment = SensorBasedSwitchingExperiment(env_type=ENV3, machine_agent=machine, human_agent=human)
    #
    # sensor_based_experiment.run_known_human_exp(human_cost=0, switching_cost=0.4, n_try=100,
    #                                             plot_name='Env3_known_sensor_based_0_0_4')
    #
    # """unknown human sensor-based state space. It may take a while (Figure 3 and 5 in the paper)"""
    # alg2_regret, greedy_regret = sensor_based_experiment.run_unknown_human_exp(n_episode=3000, human_cost=0.2,
    #                                                                            switching_cost=0.1, n_try=100,
    #                                                                            plot_epochs=[5, 500, 2999],
    #                                                                            plot_dir='epoch_plots/')
    # plot_regret(alg2_regret, greedy_regret, file_name='Env3_regret_0_2_0_1.pdf')

    """human control and number of switches. It may take a while (Figure 4 in the paper)"""
    sd_range = np.arange(6)
    human_control_rate, switch_number = change_sd(env_type=ENV2, sd_range=np.arange(6), env_num=100, human=human,
                                                  machine=machine, switching_cost=0.2, human_cost=0.1, n_try=100)
    plot_change_sd(sd_range, switch_number, y_label=r'\# of switches',
                   legend_label=r'Env.\,2, $\lambda_1=0.2, \lambda_2=0.1$',
                   file_name='Env2_0_2_0_1_switch_vs_sd.pdf')

    plot_change_sd(sd_range, human_control_rate, y_label=r'\% of human control',
                   legend_label=r'Env.\,2, $\lambda_1=0.2, \lambda_2=0.1$',
                   file_name='Env2_0_2_0_1_switch_vs_sd.pdf')
