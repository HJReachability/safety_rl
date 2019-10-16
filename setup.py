from setuptools import setup, find_packages

setup(
    name='mdr_rl',
    version='0.1',
    description='This package uses the Safety Bellman Equation with modern reinforcement learning '
                'techniques to find approximations of reachable sets and safe policies',
    author='Neil Lugovoy',
    author_email='nflugovoy@berkeley.edu',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'spinup', 'tensorflow==1.9.0', 'gym', 'scipy',
                      'ray==0.7.3', 'pandas']
)
