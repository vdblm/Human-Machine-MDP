from collections import defaultdict
import numpy as np
import pylab as pl
from environments.lane_keeping import make_default_cell_types


class PlotPath:
    """
    plot agent path in a grid experiment
    """

    def __init__(self, width, height, n_try, cell_types=None, img_dir=None, start_point=None):
        """

        :param n_try: number of repeating the experiment
        :param cell_types: type of each cell like 'road', 'grass', etc.
        :type cell_types: dict
        :param img_dir: directory of cell types images
        """
        self.width = width
        self.height = height
        self.n_try = n_try

        self.cell_types = cell_types
        self.img_dir = img_dir
        self.start_point = start_point

        # lines {(org_x, org_y, dest_x, dest_y, color): count}
        self.lines = defaultdict(int)

        if img_dir is None:
            self.img_dir = '../../cell_types/'

    def add_line(self, src_state, dst_state, color):
        """
        adds line to the plot
        :type src_state: int
        :type dst_state: int
        """
        src_y = int(src_state / self.width)
        src_x = src_state % self.width
        dst_y = int(dst_state / self.width)
        dst_x = dst_state % self.width

        self.lines[(src_x, src_y, dst_x, dst_y, color)] += 1

    def plot(self):
        lines = []
        colors = []
        width = []
        fig, ax = pl.subplots()
        for line_tuple in self.lines:
            line = [(line_tuple[0] + 0.5, line_tuple[1] + 0.5), (line_tuple[2] + 0.5, line_tuple[3] + 0.5)]
            lines.append(line)
            colors.append(line_tuple[4])
            width.append(self.lines.get(line_tuple))

        # add image
        if self.cell_types is not None:
            for x in range(self.width):
                for y in range(self.height):
                    if self.cell_types[x, y] == 'end':
                        pl.text(x + 0.5, y + 0.5, 'End', horizontalalignment='center', verticalalignment='center')
                        continue
                    img = pl.imread(self.img_dir + str(self.cell_types[x, y]) + '.png')
                    ax.imshow(img, extent=[x, x + 1, y, y + 1])

        width = np.divide(width, self.n_try * 10)

        # arrows
        for i, line in enumerate(lines):
            if line[0][0] == line[1][0]:
                dx = 0
                dy = 0.4
            else:
                slope = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
                dx = 0.4 / (np.sqrt(1 + slope * slope))
                dy = slope * dx
            ax.arrow(line[0][0] - dx / 4, line[0][1] - dy / 4, dx, dy, head_length=2.5 * width[i], width=width[i],
                     length_includes_head=True, color=colors[i])

        ax.autoscale()
        pl.grid(True, linewidth=0.2, color='black')
        ax.set_xticks([])
        ax.set_yticks([])

        # add grid lines
        ratio = 10
        x = np.linspace(0, self.width - 1, self.width * ratio)
        for y in range(self.height):
            ax.plot(x, [y for i in range(self.width * ratio)], color='black', linewidth=0.2)

        y = np.linspace(0, self.height - 1, self.height * ratio)
        for x in range(self.width):
            ax.plot([x for i in range(self.height * ratio)], y, color='black', linewidth=0.2)

        # start point
        if self.start_point is None:
            self.start_point = (2, 0)
        pl.text(self.start_point[0] + 0.5, self.start_point[1] + 0.5, 'Start', horizontalalignment='center',
                verticalalignment='center')
        return pl


def test_PlotPath():
    # test PlotPath
    cell_types = make_default_cell_types(7, 10)
    plt_path = PlotPath(7, 10, 1, cell_types=cell_types)
    plt_path.add_line(0, 7, 'orange')
    plt_path.add_line(7, 15, 'deepskyblue')
    plt_path.add_line(15, 16, 'orange')
    pl = plt_path.plot()
    pl.tight_layout()
    pl.show()


if __name__ == '__main__':
    test_PlotPath()
