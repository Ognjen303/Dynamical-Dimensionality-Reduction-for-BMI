import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def black_c(c): return LinearSegmentedColormap.from_list(
    'BlkGrn', [(0, 0, 0), c], N=256)


def get_colors(xs, ys, alt_colors=False):
    """
    returns a list of colors (that can be passed as the optional "color" or "c" input arguments to Matplotlib plotting functions)
    based on the values in the coordinate lists (or 1D array) xs and ys. More specifically, colors are based on the
    projected location along a direction with the widest spread of points.
    :param xs, ys: two vectors (lists or 1D arrays of the same length) containing the x and y coordinates of a set of points
    :param alt_colors: if True, the green and red color poles (for negative and positive values) are switched to cyan and magenta.
    :return:
    colors: a list (with same length as xs) of colors corresponding to coorinates along the maximum-spread direction:
    small values are closer to black, large positive values closer to red, and large negative values closer to green.
    The elements of "colors" can be passed as the optional "color" or "c" input argument to Matplotlib plotting functions.
    """
    xys = np.array([xs, ys])
    u, _, _ = np.linalg.svd(xys)
    def normalize(x): return 2 * (x - x.min()) / (x.max() - x.min()) - 1
    xs = normalize(u[:, 0].T @ xys)
    if alt_colors:
        pos_cmap = black_c((1, 0, 1))
        neg_cmap = black_c((0, 1, 1))
    else:
        pos_cmap = black_c((1, 0, 0))
        neg_cmap = black_c((0, 1, 0))

    colors = []
    for x in xs:
        if x < 0:
            colors.append(neg_cmap(-x))
        else:
            colors.append(pos_cmap(+x))

    return colors


def plot_start(xs, ys, colors, markersize=500, ax=None):
    """
    Puts round markers on the starting point of trajectories
    :param xs: x-coordinates of the initial point of trajectories
    :param ys: y-coordinates of the initial point of trajectories
    :param colors: colors for different conditions obtained using the get_colors function
    :param markersize: size of the markers
    :param ax: axis on which to plot (optional)
    """
    if ax is None:
        plt.scatter(xs, ys, s=markersize, color=colors,
                    marker=".", edgecolors="k")
    else:
        ax.scatter(xs, ys, s=markersize, color=colors,
                   marker=".", edgecolors="k")


def plot_end(xs, ys, colors, markersize=100, ax=None):
    """
    Puts diamond-shaped markers on the end point of trajectories
    :param xs: x-coordinates of the final point of trajectories
    :param ys: y-coordinates of the final point of trajectories
    :param colors: colors for different conditions obtained using the get_colors function
    :param markersize: size of the markers
    :param ax: axis on which to plot (optional)
    """
    if ax is None:
        plt.scatter(xs, ys, s=markersize, color=colors,
                    marker="D", edgecolors="k")
    else:
        ax.scatter(xs, ys, s=markersize, color=colors,
                   marker="D", edgecolors="k")
