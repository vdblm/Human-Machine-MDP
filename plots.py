import logging
import pickle
import numpy as np
import matplotlib as mpl

mpl.use('pdf')
import matplotlib.pyplot as plt


def latexify(fig_width=None, fig_height=None, columns=1, largeFonts=False, small_font=7):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        logging.warning("WARNING: fig_height too large:" + str(fig_height) +
                        "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    ratio = 1
    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{amsmath,amsfonts,amssymb,bbm,amsthm, mathtools,times}'],
        'axes.labelsize': 10 if largeFonts else small_font,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10 if largeFonts else small_font,
        'font.size': 10 if largeFonts else small_font,  # was 10
        'legend.fontsize': 10 if largeFonts else small_font * ratio,  # was 10
        'xtick.labelsize': 10 if largeFonts else small_font * ratio,
        'ytick.labelsize': 10 if largeFonts else small_font * ratio,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif',
        'xtick.minor.size': 0.5,
        'xtick.major.pad': 1.5,
        'xtick.major.size': 1,
        'ytick.minor.size': 0.5,
        'ytick.major.pad': 1.5,
        'ytick.major.size': 1
    }

    mpl.rcParams.update(params)
    plt.rcParams.update(params)


SPINE_COLOR = 'grey'


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


colors = ['#3E9651', '#CC2529', '#396AB1', '#535154']

# hum_env = pickl

if __name__ == '__main__':
    # column_width = 234.8775
    # width = column_width * 0.0138889 / 2
    # height = width / 1.4
    a = (np.sqrt(5) - 1.0) / 2
    width = 1.3
    height = width * a
    latexify(fig_height=height, fig_width=width, small_font=7)
    plt.rcParams['legend.fontsize'] = 5.6
    fig, ax = plt.subplots()

    ofu = pickle.load(open('outputs/unknown/ucb_regret_0.1_0.2_light', 'rb'))
    greedy = pickle.load(open('outputs/unknown/greedy_regret_0.1_0.2_light', 'rb'))
    ax.plot(np.cumsum(ofu), colors[1], label=r'Algorithm 2', linestyle='solid',
            linewidth=1)
    ax.plot(np.cumsum(greedy), colors[0], label=r'Greedy Baseline', linestyle='solid',
            linewidth=1)

    # ofu = pickle.load(open('outputs/unknown/ucb_regret_0_0_heavy', 'rb'))
    # greedy = pickle.load(open('outputs/unknown/greedy_regret_0_0_heavy', 'rb'))
    #
    # # fig.subplots_adjust(left=0, bottom=0, right=30, top=30)
    # ax.plot(np.cumsum(ofu), colors[1], label=r'Algorithm 2, $\lambda_1=0, \lambda_2=0$', linestyle='solid', linewidth=1)
    # ax.plot(np.cumsum(greedy), colors[0], label=r'Greedy Baseline, $\lambda_1=0, \lambda_2=0$', linestyle='solid',
    #         linewidth=1)

    ax.set_ylabel(r'Regret, $R(T)$')
    ax.set_xlabel(r'Episode, $k$')
    ax.legend(frameon=False)
    ax = format_axes(ax)
    # fig.set_size_inches(width, height, forward=True)
    # plt.annotate(fontsize=1)
    fig.savefig('outputs/unknown/light_12.pdf', bbox_inches='tight')

# LaTeX
