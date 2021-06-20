# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import numpy as np

def plot_arc(center, r, thetaParam, ax, c='b', lw=1.5, orientation=0,
    zorder=0):  
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

    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)

    theta = np.linspace(thetaInit+orientation, thetaFinal+orientation, 100)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)

    ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_arc(center, r, thetaParam, ax, c='b', lw=1.5, orientation=0):
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
    """
    x, y = center
    thetaInit, thetaFinal = thetaParam

    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)

    theta = np.linspace(thetaInit+orientation, thetaFinal+orientation, 100)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)

    ax.plot(xs, ys, c=c, lw=lw)


def plot_circle(center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False, zorder=0):
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
    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)

    theta = np.linspace(0, 2*np.pi, 200)
    xs = xtilde + r * np.cos(theta)
    ys = ytilde + r * np.sin(theta)
    ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
    if scatter:
        ax.scatter(xtilde+r, ytilde, c=c, s=80)
        ax.scatter(xtilde-r, ytilde, c=c, s=80)
        print(xtilde+r, ytilde, xtilde-r, ytilde)


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
    xtilde = x*np.cos(orientation) - y*np.sin(orientation)
    ytilde = y*np.cos(orientation) + x*np.sin(orientation)
    thetatilde = theta+orientation

    return np.array([xtilde, ytilde, thetatilde])