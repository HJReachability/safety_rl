"""
Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

from setuptools import setup

setup(name="gym_reachability",
      version="0.0.1",
      install_requires=["gym", "numpy", "math", "pyglet", "Box2D", "sys", "mujoco_py"])
