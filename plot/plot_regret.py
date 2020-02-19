import os

import numpy as np
import matplotlib as mpl

mpl.use('pdf')
import matplotlib.pyplot as plt
from plot.plot_path import OUTPUT_DIR


def latexify(fig_width, fig_height, font_size=7, legend_size=5.6):
    """Set up matplotlib's RC params for LaTeX plotting."""
    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{amsmath,amsfonts,amssymb,bbm,amsthm, mathtools,times}'],
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        'legend.fontsize': legend_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
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


COLORS = ['#3E9651', '#CC2529', '#396AB1', '#535154']
golden_ratio = (np.sqrt(5) - 1.0) / 2


def plot_regret(alg2_regret, greedy_regret, file_name):
    width = 1.3
    height = width * golden_ratio

    latexify(fig_height=height, fig_width=width, font_size=5, legend_size=4)

    fig, ax = plt.subplots()
    ax = format_axes(ax)

    ax.plot(np.cumsum(alg2_regret), COLORS[1], label=r'Algorithm 2', linestyle='solid',
            linewidth=1)
    ax.plot(np.cumsum(greedy_regret), COLORS[0], label=r'Greedy Baseline', linestyle='solid',
            linewidth=1)

    ax.set_ylabel(r'Regret, $R(T)$')
    ax.set_xlabel(r'Episode, $k$')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUTPUT_DIR, file_name), bbox_inches='tight')


def plot_change_sd(sd_range, y, y_label, legend_label, file_name):
    height = 1.5
    width = height / golden_ratio
    latexify(fig_height=height, fig_width=width, font_size=5, legend_size=3.9)
    fig, ax = plt.subplots()
    ax = format_axes(ax)
    line_width = 1
    ax.plot(sd_range, y, COLORS[0], label=legend_label, linestyle='solid',
            linewidth=line_width, marker='d', ms=2)

    ax.set_ylabel(y_label)
    ax.set_xlabel(r'$\sigma_{\mathbb{H}}$')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUTPUT_DIR, file_name), bbox_inches='tight')
