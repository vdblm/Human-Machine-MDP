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


def regularize(hum_env):
    x = list(zip(*hum_env['sensor']))[0]
    y = list(zip(*hum_env['sensor']))[1]
    x = np.round(x, decimals=2)
    xx = [i for i in range(101)]
    yy = [0 for i in range(101)]
    cntr = [0 for i in range(101)]
    for j, i in enumerate(x):
        # print(i * 100)
        cntr[int(i * 100)] += 1
        yy[int(i * 100)] += y[j]

    new_x = []
    new_y = []
    for i in range(101):
        if cntr[i] != 0:
            yy[i] /= cntr[i]
            new_x.append(xx[i])
            new_y.append(yy[i])
    return new_x, new_y


def regularizer2(switch_env):
    x = list(zip(*switch_env['sensor']))[0]
    y = list(zip(*switch_env['sensor']))[1]
    x = np.round(x, decimals=0)
    m = int(max(x) + 1)
    xx = [i for i in range(0, m)]
    yy = [0 for i in range(m)]
    cntr = [0 for i in range(m)]
    for j, i in enumerate(x):
        cntr[int(i)] += 1
        yy[int(i)] += y[j]

    new_x = []
    new_y = []
    for i in range(m):
        if cntr[i] != 0:
            yy[i] /= cntr[i]
            new_x.append(xx[i])
            new_y.append(yy[i])
    return new_x, new_y


def reg3(var_env):
    tmp = list(zip(*var_env))
    switch_n = tmp[0]
    hum_cntr = tmp[1]
    env_cost = tmp[2]
    var = tmp[3]
    xx = np.arange(0, 6, 1)
    switch_y = [0 for i in range(6)]
    human_y = [0 for i in range(6)]
    env_y = [0 for i in range(6)]
    cntr = [0 for i in range(6)]
    for i, j in enumerate(var):
        if np.abs(j // 1 - j) > 0.1 or j > 5.1:
            continue
        index = int(j)
        cntr[index] += 1
        switch_y[index] += switch_n[i]
        human_y[index] += hum_cntr[i]
        env_y[index] += env_cost[i]
    for i in range(6):
        switch_y[i] /= cntr[i]
        human_y[i] /= cntr[i]
        env_y[i] /= cntr[i]
    print(cntr)
    return xx, switch_y, np.multiply(human_y, 100), env_y


colors = ['#3E9651', '#CC2529', '#396AB1', '#535154']

# hum_env = pickle.load(open('outputs/env1_hum_env', 'rb'))
# print(len(hum_env['sensor']))
# x, y = regularize(hum_env)

var_env = pickle.load(open('outputs/env2_var_env_00', 'rb'))
x2, switch2, hum2, env2 = reg3(var_env)

var_env = pickle.load(open('outputs/env1_var_env_00', 'rb'))
x1, switch1, hum1, env1 = reg3(var_env)

height = 1.5
width = height / ((np.sqrt(5) - 1.0) / 2.0)
latexify(fig_height=height, fig_width=width, small_font=5)
plt.rcParams['legend.fontsize'] = 3.9
fig, ax = plt.subplots()
linewidth = 1
ax.plot(x1, switch1, colors[0], label=r'Env.\,1, $\lambda_1=0, \lambda_2=0$', linestyle='solid', linewidth=linewidth,
        marker='d', ms=2)
ax.plot(x2, switch2, colors[1], label=r'Env.\,2, $\lambda_1=0, \lambda_2=0$', linestyle='solid', linewidth=linewidth,
        marker='d', ms=2)
var_env = pickle.load(open('outputs/env2_var_env', 'rb'))
x2, switch2, hum2, env2 = reg3(var_env)

var_env = pickle.load(open('outputs/env1_var_env', 'rb'))
x1, switch1, hum1, env1 = reg3(var_env)
ax.plot(x1, switch1, colors[2], label=r'Env.\,1, $\lambda_1=0.2, \lambda_2=0.1$', linestyle='solid',
        linewidth=linewidth, marker='d', ms=2)
ax.plot(x2, switch2, colors[3], label=r'Env.\,2, $\lambda_1=0.2, \lambda_2=0.1$', linestyle='solid',
        linewidth=linewidth, marker='d', ms=2)
ax.set_ylabel(r'\# of switches')
ax.set_xlabel(r'$\sigma_{\mathbb{H}}$')
ax.legend(frameon=False, loc='lower left')
ax = format_axes(ax)
fig.savefig('outputs/switch_var_total', bbox_inches='tight')
