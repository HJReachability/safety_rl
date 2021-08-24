"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
import numpy as np


# == margin ==
def calculate_margin_rect(s, x_y_w_h, negativeInside=True):
    """
    _calculate_margin_rect: calculate the margin to the box.

    Args:
        s (np.ndarray): the state.
        x_y_w_h (box specification): (center_x, center_y, width, height).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
    x, y, w, h = x_y_w_h
    delta_x = np.abs(s[0] - x)
    delta_y = np.abs(s[1] - y)
    margin = max(delta_y - h/2, delta_x - w/2)

    if negativeInside:
        return margin
    else:
        return -margin


def calculate_margin_circle(s, c_r, negativeInside=True):
    """
    _calculate_margin_circle: calculate the margin to the circle.

    Args:
        s (np.ndarray): the state.
        c_r (circle specification): (center, radius).
        negativeInside (bool, optional): add a negative sign to the distance
            if inside the box. Defaults to True.

    Returns:
        float: margin.
    """
    center, radius = c_r
    dist_to_center = np.linalg.norm(s[:2] - center)
    margin = dist_to_center - radius

    if negativeInside:
        return margin
    else:
        return -margin


# == Plotting ==
def plot_arc(
    center, r, thetaParam, ax, c='b', lw=1.5, orientation=0, zorder=0
):
    """
    plot_arc

    Args:
        center (np.ndarray): center.
        r (float): radius.
        thetaParam (np.ndarray): [thetaInit, thetaFinal].
        ax (matplotlib.axes.Axes)
        c (str, optional): color. Defaults to 'b'.
        lw (float, optional): linewidth. Defaults to 1.5.
        orientation (int, optional): counter-clockwise angle. Defaults to 0.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    x, y = center
    thetaInit, thetaFinal = thetaParam

    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)

    theta = np.linspace(thetaInit + orientation, thetaFinal + orientation, 100)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)

    ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False,
    zorder=0
):
    """
    plot_circle

    Args:
        center (np.ndarray): center.
        r (float): radius.
        ax (matplotlib.axes.Axes)
        c (str, optional): color. Defaults to 'b'.
        lw (float, optional): linewidth. Defaults to 1.5.
        ls (str, optional): linestyle. Defaults to '-'.
        orientation (int, optional): counter-clockwise angle. Defaults to 0.
        scatter (bool, optional): show center or not. Defaults to False.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    x, y = center
    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)

    theta = np.linspace(0, 2 * np.pi, 200)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)
    ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
    if scatter:
        ax.scatter(xtilde + r, ytilde, c=c, s=80)
        ax.scatter(xtilde - r, ytilde, c=c, s=80)
        print(xtilde + r, ytilde, xtilde - r, ytilde)


def rotatePoint(state, orientation):
    """
    rotatePoint

    Args:
        state (np.ndarray): (x, y) position.
        orientation (int, optional): counter-clockwise angle.

    Returns:
        np.ndarray: rotated state.
    """
    x, y, theta = state
    xtilde = x * np.cos(orientation) - y * np.sin(orientation)
    ytilde = y * np.cos(orientation) + x * np.sin(orientation)
    thetatilde = theta + orientation

    return np.array([xtilde, ytilde, thetatilde])
