import numpy as np


class EnvironmentType:
    """
    Different environment types in the paper
    """

    def __init__(self, type_probs: dict, mid_lane_type_probs: dict = None):
        """
        Parameters
        ----------
        type_probs : dict[str, float]
            Probability of each cell type. For example: {'road': 0.5, 'stone': 0.3, ...}
        mid_lane_type_probs: None or dict[str, float]
            Probability of cell types in the middle lane. If `mid_lane_type_probs=None`,
            then the probabilities of the middle cell = `type_probs`

        """
        self.type_probs = type_probs
        self.mid_lane_type_probs = mid_lane_type_probs

    def generate_cell_types(self, width, height):
        """
        Assign each cell a type (i.e., 'road', 'grass', 'stone', or 'car')
        independently at random with the given probabilities.

        Parameters
        ----------
        width : int
            Width of the grid env
        height : int
            Height of the grid env

        Returns
        -------
        cell_types : dict
             Type of each grid cell with coordinate (x, y), i.e., {(x, y): type}.
        """

        def __in_middle(cell):
            """ checks if the cell is in the middle line"""
            x = cell[0]
            if np.floor((width - 1) / 2) <= x <= np.ceil((width - 1) / 2):
                return True
            return False

        cell_types = {}
        types = list(self.type_probs.keys())
        probs = list(self.type_probs.values())

        if self.mid_lane_type_probs is not None:
            mid_types = list(self.mid_lane_type_probs.keys())
            mid_probs = list(self.mid_lane_type_probs.values())

        for x in range(width):
            for y in range(height):
                if self.mid_lane_type_probs is not None and __in_middle(cell=(x, y)):
                    cell_types[x, y] = np.random.choice(mid_types, p=mid_probs)
                else:
                    cell_types[x, y] = np.random.choice(types, p=probs)
        return cell_types


# default environment types
ENV1 = EnvironmentType(type_probs={'road': 0.4, 'grass': 0.3, 'stone': 0.3},
                       mid_lane_type_probs={'road': 1, 'grass': 0, 'stone': 0, 'car': 0})

ENV2 = EnvironmentType(type_probs={'road': 0.4, 'grass': 0.3, 'stone': 0.3, 'car': 0},
                       mid_lane_type_probs={'road': 0.6, 'grass': 0, 'stone': 0, 'car': 0.4})

ENV3 = EnvironmentType(type_probs={'road': 0.5, 'grass': 0.3, 'car': 0.2},
                       mid_lane_type_probs=None)

ENV3_LIGHT_TRAFFIC = EnvironmentType(type_probs={'road': 0.7, 'grass': 0.2, 'car': 0.1},
                                     mid_lane_type_probs=None)
