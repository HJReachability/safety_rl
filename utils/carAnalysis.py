"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

import numpy as np

# == Color Param ==
brown = "#8B4513"
purple = "#9370DB"
tiffany = "#0abab5"
pink = "#FFB6C1"
silver = "#C0C0C0"


def thetaMtx(theta):
  """Returns the rotation matrix given an angle.

  Args:
      theta (float): the angle to rotate.
  """
  return np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])


# == Basic Plots ==
def plot_arc(p, r, thetaParam, ax, c="b", lw=3, orientation=0., zorder=1):
  """Plots an arc given a center, a radius and the (theta_init, theta_final).

  Args:
      p (np.ndarray): the center of the arc.
      r (float): the radius of the arc.
      thetaParam (tuple of floats): the initial angle and the final angle.
      ax (matplotlib.axes.Axes): the ax to plot.
      c (str, optional): the color of the arc. Defaults to "b".
      lw (int, optional): the linewidth of the arc. Defaults to 3.
      orientation (float, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      zorder (int, optional): the z-order of the plot. Defaults to 1.
  """
  x, y = p
  theta_init, theta_final = thetaParam

  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(theta_init + orientation, theta_final + orientation, 100)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)

  ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    x, y, r, ax, c="b", lw=5, orientation=0., scatter=False, zorder=1
):
  """Plots a circle given a center and a radius.

  Args:
      x (float): the x position of the circle.
      y (float): the y position of the circle.
      r (float): the radius of the circle.
      ax (matplotlib.axes.Axes): the ax to plot.
      c (str, optional): the color of the circle. Defaults to "b".
      lw (int, optional): the linewidth of the circle. Defaults to 5.
      orientation (float, optional): rotate points in the xy-plane
          counterclockwise through orientation with respect to the x axis.
          Defaults to 0.
      scatter (bool, optional): Show two point on the circle if True.
          Defaults to False.
      zorder (int, optional): the z-order of the plot. Defaults to 1.
  """
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(0, 2 * np.pi, 200)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)
  ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)
  if scatter:
    ax.scatter(xtilde + r, ytilde, c=c, s=80, zorder=zorder)
    ax.scatter(xtilde - r, ytilde, c=c, s=80, zorder=zorder)
    print(
        "Intersection Points: ({0}, {1}) | ({2}, {1})".format(
            xtilde + r, ytilde, xtilde - r
        )
    )


# == Outer Safe Set ==
def plot_outer_safety(R, R_turn, orientation, ax, extent, lw=3):
  """
  Plots the safe set considering the outer circle (the boundary of the failure
  set).

  Args:
      R (float): the radius of the outer circle.
      R_turn (float): the radius of the car's circular motion.
      orientation (float): rotate points in the xy-plane clockwise through
          orientation with respect to the x axis. Here it is clockwise because
          we are using body-oriented coordinate.
      ax (matplotlib.axes.Axes): the ax to plot.
      extent (list of 4 floats): the bounding box in data coordinates that the
          image will fill.
      lw (int, optional): the linewidth of the set boundary. Defaults to 3.
  """
  nx, ny = 300, 300
  xs = np.linspace(-R, R, nx)
  ys = np.linspace(-R, R, ny)
  # print(xs)
  v = np.full((nx, ny), fill_value=False)

  it = np.nditer(v, flags=["multi_index"])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    xtilde = x * np.cos(orientation) + y * np.sin(orientation)
    ytilde = y * np.cos(orientation) - x * np.sin(orientation)

    boolIn = (x**2 + y**2) <= R**2
    if np.abs(ytilde) > 2*R_turn - R:
      bool0 = xtilde <= np.sqrt((R - R_turn)**2 - (R_turn - np.abs(ytilde))**2)
    else:
      bool0 = False
    if np.abs(ytilde) <= 2*R_turn - R:
      bool1 = xtilde <= -1 * np.sqrt((3*R_turn - R)**2 -
                                     (R_turn + np.abs(ytilde))**2)
    else:
      bool1 = False

    v[idx] = not ((bool0 or bool1) and boolIn)
    it.iternext()

  ax.imshow(
      v.T, interpolation="none", extent=extent, origin="lower", cmap="plasma",
      vmin=0
  )

  tmpTheta = np.arccos(R_turn / (3*R_turn - R))
  plot_arc((0.0, R_turn), R - R_turn, (-np.pi / 2, np.pi / 2), ax, c=tiffany,
           lw=lw, orientation=orientation)
  plot_arc((0.0, -R_turn), R - R_turn, (-np.pi / 2, np.pi / 2), ax, c=tiffany,
           lw=lw, orientation=orientation)
  plot_arc((0.0, -R_turn), 3*R_turn - R, (np.pi / 2, np.pi / 2 + tmpTheta), ax,
           c=tiffany, lw=lw, orientation=orientation)
  plot_arc((0.0, R_turn), 3*R_turn - R, (-np.pi / 2, -np.pi / 2 - tmpTheta),
           ax, c=tiffany, lw=lw, orientation=orientation)
  plot_arc((0.0, 0), R, (np.pi / 2, 3 * np.pi / 2), ax, c=tiffany, lw=lw,
           orientation=orientation)
  return


# == Reach-Avoid Set ==
def plot_reach_avoid_type_1(
    R, R_turn, r, orientation, ax, extent, fig=None, cbarPlot=False, zorder=1,
    lw=3, cmap="seismic"
):
  """
  Plots the reach-avoid set for the environment that satisfies the radius of
  inner circle is no smaller than 2 times of the radius of circular motion
  minus the radius of the outer circle, i.e., r >= 2*R_turn-R.

  Args:
      R (float): the radius of the outer circle
      R_turn (float): the radius of circular motion
      r (float): the radius of inner circle
      orientation (float): otate points in the xy-plane clockwise through
          orientation with respect to the x axis. Here it is clockwise because
          we are using body-oriented coordinate.
      ax (matplotlib.axes.Axes): the ax to plot.
      extent (list of 4 floats): the bounding box in data coordinates that the
          image will fill.
      fig (matplotlib.Figure, optional): the figure to plot. Defaults to None.
      cbarPlot (bool, optional): plot the colorbar if True. Defaults to False.
      zorder (int, optional): the z-order of the plot. Defaults to 1.
      lw (int, optional): the linewidth of the set boundary. Defaults to 3.
      cmap (str, optional): the colormap. Defaults to "seismic".

  Returns:
      float: the area of the reach-avoid set.
  """
  nx, ny = 300, 300
  xs = np.linspace(extent[0], extent[1], nx)
  ys = np.linspace(extent[2], extent[3], ny)

  v = np.full((nx, ny), fill_value=False)
  it = np.nditer(v, flags=["multi_index"])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    xtilde = x * np.cos(orientation) + y * np.sin(orientation)
    ytilde = y * np.cos(orientation) - x * np.sin(orientation)

    boolIn = (x**2 + y**2) <= R**2
    if np.abs(ytilde) > 2*R_turn - R:
      bool0 = xtilde <= np.sqrt((R - R_turn)**2 - (R_turn - np.abs(ytilde))**2)
    else:
      bool0 = False
    if np.abs(ytilde) <= r:
      bool1 = xtilde <= np.sqrt(r**2 - ytilde**2)
    else:
      bool1 = False

    v[idx] = not ((bool0 or bool1) and boolIn)
    it.iternext()
  im = ax.imshow(
      v.T, interpolation="none", extent=extent, origin="lower", cmap=cmap,
      vmin=0, zorder=-1
  )
  if cbarPlot:
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=0.95, ticks=[0, 1]
    )
    cbar.ax.set_yticklabels(labels=[0, 1], fontsize=16)
  # plot arc
  tmpY = (r**2 - R**2 + 2*R_turn*R) / (2*R_turn)
  tmpX = np.sqrt(r**2 - tmpY**2)
  tmpTheta = np.arcsin(tmpX / (R-R_turn))
  plot_arc((0.0, R_turn), R - R_turn, (tmpTheta - np.pi / 2, np.pi / 2), ax,
           c="g", lw=lw, orientation=orientation, zorder=zorder)
  plot_arc((0.0, -R_turn), R - R_turn, (-np.pi / 2, np.pi / 2 - tmpTheta), ax,
           c="g", lw=lw, orientation=orientation, zorder=zorder)
  plot_arc((0.0, 0), R, (np.pi / 2, 3 * np.pi / 2), ax, c="g", lw=lw,
           orientation=orientation, zorder=zorder)
  tmpPhi = np.arcsin(tmpX / r)
  plot_arc((0.0, 0), r, (tmpPhi - np.pi / 2, np.pi / 2 - tmpPhi), ax, c="g",
           lw=lw, orientation=orientation, zorder=zorder)

  areaY = 0.5 * (R - R_turn)**2 * (np.pi - tmpTheta)
  areaG = 0.5 * r**2 * (np.pi / 2 - tmpPhi)
  areaB = 0.5 * R_turn * tmpX

  return 2 * (areaY+areaG+areaB) + 0.5 * R**2 * np.pi


def plot_reach_avoid_type_2(
    R, R_turn, r, orientation, ax, extent, fig=None, cbarPlot=False, zorder=1,
    lw=3, cmap="seismic"
):
  """
  Plots the reach-avoid set for the environment that satisfies the radius of
  inner circle is smaller than 2 times of the radius of circular motion
  minus the radius of the outer circle, i.e., r < 2*R_turn-R.

  Args:
      R (float): the radius of the outer circle
      R_turn (float): the radius of circular motion
      r (float): the radius of inner circle
      orientation (float): otate points in the xy-plane clockwise through
          orientation with respect to the x axis. Here it is clockwise because
          we are using body-oriented coordinate.
      ax (matplotlib.axes.Axes): the ax to plot.
      extent (list of 4 floats): the bounding box in data coordinates that the
          image will fill.
      fig (matplotlib.Figure, optional): the figure to plot. Defaults to None.
      cbarPlot (bool, optional): plot the colorbar if True. Defaults to False.
      zorder (int, optional): the z-order of the plot. Defaults to 1.
      lw (int, optional): the linewidth of the set boundary. Defaults to 3.
      cmap (str, optional): the colormap. Defaults to "seismic".
  """
  nx, ny = 300, 300
  xs = np.linspace(extent[0], extent[1], nx)
  ys = np.linspace(extent[2], extent[3], ny)
  # print(xs)
  v = np.full((nx, ny), fill_value=False)

  it = np.nditer(v, flags=["multi_index"])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    xtilde = x * np.cos(orientation) + y * np.sin(orientation)
    ytilde = y * np.cos(orientation) - x * np.sin(orientation)

    boolIn = (x**2 + y**2) <= R**2
    if np.abs(ytilde) > r:
      bool0 = xtilde <= -np.sqrt((R_turn - r)**2 -
                                 (R_turn - np.abs(ytilde))**2)
    else:
      bool0 = False
    if np.abs(ytilde) <= r:
      bool1 = xtilde <= np.sqrt((r**2 - ytilde**2))
    else:
      bool1 = False

    v[idx] = not ((bool0 or bool1) and boolIn)
    it.iternext()
  im = ax.imshow(
      v.T, interpolation="none", extent=extent, origin="lower", cmap=cmap,
      vmin=0, zorder=-1
  )
  if cbarPlot:
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=0.95, ticks=[0, 1]
    )
    cbar.ax.set_yticklabels(labels=[0, 1], fontsize=16)
  # two sides
  tmpY = (R**2 + 2*R_turn*r - r**2) / (2*R_turn)
  tmpX = np.sqrt(R**2 - tmpY**2)
  tmpTheta = np.arcsin(tmpX / (R_turn-r))
  tmpTheta2 = np.arcsin(tmpX / R)
  plot_arc((0.0, R_turn), R_turn - r, (np.pi / 2 + tmpTheta, 3 * np.pi / 2),
           ax, c="g", lw=lw, orientation=orientation, zorder=zorder)
  plot_arc((0.0, -R_turn), R_turn - r, (np.pi / 2, 3 * np.pi / 2 - tmpTheta),
           ax, c="g", lw=lw, orientation=orientation, zorder=zorder)
  # middle
  plot_arc((0.0, 0), r, (np.pi / 2, -np.pi / 2), ax, c="g", lw=lw,
           orientation=orientation, zorder=zorder)
  # outer boundary
  plot_arc((0.0, 0), R, (np.pi / 2 + tmpTheta2, 3 * np.pi / 2 - tmpTheta2), ax,
           c="g", lw=lw, orientation=orientation, zorder=zorder)
