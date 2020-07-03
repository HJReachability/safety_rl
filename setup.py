"""
Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

See the LICENSE in the root directory of this repo for license info.
"""

from setuptools import setup, find_packages

setup(
    name='safety_rl',
    version='0.2',
    description=['This package uses the Safety Bellman Equation with modern',
                 'reinforcement learning techniques to find approximations of',
                 'safe sets and safe policies.'],
    author='Neil Lugovoy',
    author_email='nflugovoy@berkeley.edu',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'spinup', 'tensorflow==1.9.0',
                      'gym', 'scipy', 'requests', 'ray==0.7.3', 'pandas',
                      'opencv-python', 'psutil', 'lz4', 'Box2D', 'mujoco-py']
)
