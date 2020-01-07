"""
implementation of known human experiment in the paper

author: balazadehvahid@gmail.com
"""
from agents.beliefs import FixedBelief
from agents.switching_agents import *
from grid_plot.plot_path import PlotPath
from environments.lane_keeping import make_lane_keeping, make_default_lane_keeping_cell_types
from agents.lane_keeping_agent import LaneKeepingAgent


def run_known_human_exp(width, height, start_point, n_try, cell_types, costs, human_agent, machine_agent, env_make_func,
                        switching_cost, non_default_agent_cost, default_agent='machine', verbose=True,
                        human_color='orange', machine_color='deepskyblue',
                        output_dir='outputs/lane_keeping_known/arrows/'):
    env, cost_true, transition_true = env_make_func(width, height, costs, cell_types, start_point)
    plt_path = PlotPath(width, height, n_try, start_point, cell_types, img_dir='cell_types/')
    transition_prior = FixedBelief(transition_true)
    cost_prior = FixedBelief(cost_true)
    human_prior = FixedBelief(human_agent)
    machine_prior = FixedBelief(machine_agent)

    agents = {'human', 'machine'}
    agents.remove(default_agent)
    non_default_agent = agents.pop()

    # agent0 is the default agent
    agent0, agent1 = eval(default_agent + '_agent'), eval(non_default_agent + '_agent')
    agent0_prior, agent1_prior = eval(default_agent + '_prior'), eval(non_default_agent + '_prior')
    clr0, clr1 = eval(default_agent + '_color'), eval(non_default_agent + '_color')

    switching_agent = UCB(env, agent0_prior, agent1_prior, cost_prior, transition_prior, switching_cost, 0,
                          non_default_agent_cost)

    for i in range(n_try):
        finished = False
        env.reset()
        switching_agent.update_policy()
        curr_agent = -1
        curr_state = env.state
        if verbose:
            print("Try #{}".format(i))
        while not finished:
            prev_agent = curr_agent
            prev_state = curr_state
            curr_agent = switching_agent.take_action(curr_state, prev_agent, env.time_step)
            action = None
            if curr_agent == 0:
                action = agent0.take_action(env.state)
            elif curr_agent == 1:
                action = agent1.take_action(env.state)

            curr_state, cost, finished = env.step_forward(action)
            switching_agent.update_obs(prev_state, curr_agent, action, cost, curr_state, finished)

            c = clr0
            if curr_agent == 1:
                c = clr1

            plt_path.add_line(prev_state, curr_state, c)
            if verbose:
                print("t #{} (s #{}, agent #{}, a #{}) --> (s #{}, c #{})".format(env.time_step - 1,
                                                                                  prev_state,
                                                                                  curr_agent,
                                                                                  action,
                                                                                  curr_state,
                                                                                  cost))
    env.reset()
    pl = plt_path.plot(arrow_plot=True)
    pl.tight_layout()
    pl.margins(0, 0)
    pl.savefig(output_dir +
               str(switching_cost).replace('.', '_') + '_' + str(non_default_agent_cost).replace('.', '_') + '.png',
               bbox_inches='tight',
               pad_inches=0)
    pl.close()


if __name__ == '__main__':
    # default is machine in lane keeping
    switch_costs = [0, 0.01, 0.1]
    human_costs = [0, 0.05, 0.1]

    width, height, start_point = 7, 10, (3, 0)
    n_try = 1
    cell_types = make_default_lane_keeping_cell_types(width, height)
    costs = {'road': 0, 'grass': 0.2, 'stone': 0.4}
    human_agent = LaneKeepingAgent(width, height, cell_types, prob_weights={'stone': 0, 'grass': .05, 'road': 0.95})
    machine_agent = LaneKeepingAgent(width, height, cell_types,
                                     prob_weights={'stone': .1, 'grass': .2, 'road': .7})
    for s in switch_costs:
        for h in human_costs:
            run_known_human_exp(width, height, start_point, n_try, cell_types, costs, human_agent, machine_agent,
                                env_make_func=make_lane_keeping, switching_cost=s, non_default_agent_cost=h,
                                verbose=False)
