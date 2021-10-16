# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

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
