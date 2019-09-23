"""
implementation of known human experiment in the paper

author: balazadehvahid@gmail.com
"""

from agent import *
from environment import *


def run_known_human_exp(switching_cost, human_cost, episodes=100, verbose=True):
    plt_path = PlotPath(episodes=episodes)
    chessboard, cost_true, transition_true = make_chessboard()
    machine = ChessboardMachineAgent(chessboard)
    human = ChessboardHumanAgent(chessboard)

    machine_prior = FixedBelief(machine)
    human_prior = FixedBelief(human)

    transition_prior = FixedBelief(transition_true)
    cost_prior = FixedBelief(cost_true)

    switching_agent = ExactPSRL(chessboard, machine_prior, human_prior, cost_prior, transition_prior,
                                switching_cost, human_cost)
    for i in range(episodes):
        finished = False
        chessboard.reset()
        switching_agent.update_policy()
        curr_agent = -1
        curr_state = chessboard.state
        if verbose:
            print("Episode {} starts\n time: (current state, current controller, action) --> (new state, cost) ".format(
                i))
        while not finished:
            prev_agent = curr_agent
            prev_state = curr_state
            curr_agent = switching_agent.take_action(curr_state, prev_agent, chessboard.time_step)
            action = None
            if curr_agent == 0:
                action = machine.take_action(chessboard.state)
            elif curr_agent == 1:
                action = human.take_action(chessboard.state)

            curr_state, cost, finished = chessboard.step_forward(action)
            switching_agent.update_obs(prev_state, curr_agent, action, cost, curr_state, finished)

            # human color = red, machine color = blue
            c = 'blue'
            if curr_agent == 1:
                c = 'red'

            plt_path.add_line(prev_state, curr_state, c)
            if verbose:
                print(
                    "{} ({}, {}, {}) --> ({}, {})".format(chessboard.time_step - 1, prev_state, curr_agent, action,
                                                          curr_state,
                                                          cost))
    pl, fig = plt_path.plot()
    fig.savefig('./outputs/pic' + str(switching_cost) + '_' + str(human_cost) + '.png')


if __name__ == '__main__':
    switch_costs = [0.025, 0.075, 0.125]
    human_costs = [0.0, 0.04, 0.08]
    for s in switch_costs:
        for h in human_costs:
            run_known_human_exp(s, h, verbose=False)
