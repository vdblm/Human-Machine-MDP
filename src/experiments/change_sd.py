""" Figure 4 in the paper (Number of switches and amount of human control)"""

from agents.hum_mac_agent import NoisyDriverAgent
from environments.env_types import EnvironmentType
from experiments.known_unknown_experiment import SensorBasedSwitchingExperiment


def change_sd(env_type: EnvironmentType, sd_range, env_num: int, human: NoisyDriverAgent,
              machine: NoisyDriverAgent, switching_cost: float, human_cost: float, n_try: int,
              verbose: str = True):
    """
    Parameters
    ----------
    sd_range : list or ndarray or range
        Range of standard deviation for human action policy
    env_num : int
        Number of different individual environment
    n_try : int
        Number of repeating the experiment for each individual environment

    Returns
    -------
    human_control_rate : list
        A list of average human control percentage for each sd value
    switch_number : list
        A list of average number of switches for each sd value
    """

    human_control_rate = []
    switch_number = []
    for sd in sd_range:
        human.noise_sd = sd
        total_switches = 0
        total_human_control = 0
        for i in range(env_num):
            experiment = SensorBasedSwitchingExperiment(env_type=env_type, machine_agent=machine, human_agent=human)
            result = experiment.run_known_human_exp(human_cost=human_cost, switching_cost=switching_cost, n_try=n_try,
                                                    plot=False)
            total_switches += result['exp_switch']
            total_human_control += result['human_control_rate']

        avg_human_percnt = total_human_control / env_num * 100
        human_control_rate.append(avg_human_percnt)
        avg_switch_num = total_switches / env_num
        switch_number.append(avg_switch_num)
        if verbose:
            print("################### SD: {} ###################".format(sd))
            print('average human percentage: {}'.format(avg_human_percnt))
            print('average number of switches: {}'.format(avg_switch_num))

    return human_control_rate, switch_number
