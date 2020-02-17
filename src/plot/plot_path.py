from collections import defaultdict
import os
import numpy as np
import pylab as pl
from matplotlib import collections as mc
from environments.episodic_mdp import GridMDP

from definitions import ROOT_DIR

IMG_DIR = os.path.join(ROOT_DIR, 'cell_types/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output/')

HUMAN_COLOR = 'orange'
MACHINE_COLOR = 'deepskyblue'


class PlotPath:
    """
    Plot an agent's trajectory in a grid-based environment during evaluation
    """

    def __init__(self, env: GridMDP, n_try: int, img_dir: str = IMG_DIR):
        """
        Parameters
        ----------
        env : GridMDP
            A cell-based episodic MDP that the agent interacts with
        n_try : int
            The number of repeats (i.e., the number of trajectories)
        img_dir : str
            The directory of different cell type images

        """
        self.width = env.width
        self.height = env.height
        self.n_try = n_try

        self.cell_types = env.cell_types
        self.img_dir = img_dir

        # lines {(org_x, org_y, dst_x, dst_y, color): count}
        self.lines = defaultdict(int)

    def add_line(self, src_state: int, dst_state: int, color: str):
        """
        Add line to the plot

        Parameters
        ----------
        src_state: int
            The start point (state) of the line
        dst_state: int
            The end point (state) of the line
        color : str
            The color of the line
        """
        src_y = src_state // self.width
        src_x = src_state % self.width
        dst_y = dst_state // self.width
        dst_x = dst_state % self.width

        self.lines[(src_x, src_y, dst_x, dst_y, color)] += 1

    def plot(self, file_name: str):
        """
        Plot the result

        Parameters
        ----------
        file_name : str
            Name of the output plot
        """
        lines = []
        colors = []
        widths = []
        for line_tuple, count in self.lines.items():
            line = [(line_tuple[0] + 0.5, line_tuple[1] + 0.5), (line_tuple[2] + 0.5, line_tuple[3] + 0.5)]
            lines.append(line)
            colors.append(line_tuple[4])
            widths.append(count)

        widths = np.divide(widths, self.n_try / 5)
        lc = mc.LineCollection(lines, colors=colors, linewidths=widths)

        fig, ax = pl.subplots()
        # add images
        for x in range(self.width):
            for y in range(self.height):
                cell_type = str(self.cell_types[x, y])
                # TODO only 'png'?
                img = pl.imread(self.img_dir + cell_type + '.png')
                ax.imshow(img, extent=[x, x + 1, y, y + 1])

        ax.add_collection(lc)
        ax.autoscale()
        pl.grid(True, linewidth=0.2, color='black')
        ax.set_xticks([])
        ax.set_yticks([])

        # add grid lines
        ratio = 10
        x = np.linspace(0, self.width, self.width * ratio)
        for y in range(self.height):
            ax.plot(x, [y for i in range(self.width * ratio)], color='gray', linewidth=0.2)

        y = np.linspace(0, self.height, self.height * ratio)
        for x in range(self.width):
            ax.plot([x for i in range(self.height * ratio)], y, color='gray', linewidth=0.2)

        # plot
        pl.tight_layout()
        pl.margins(0, 0)
        # TODO only png?
        pl.savefig(OUTPUT_DIR + file_name + '.png', bbox_inches='tight', pad_inches=0)
        pl.close()

        # clear `self.lines`
        self.lines = defaultdict(int)
