from setuptools import setup

setup(
    name='uclasm',
    version='1.0',
    description='Module for Subgraph Isomorphism',
    author='Jacob Moorman, Thomas Tu, Xie He, Qinyi Chen',
    packages=['uclasm', 'uclasm.utils', 'uclasm.filters', 'uclasm.counting',
              'uclasm.equivalence'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'networkx', 'ipython']
)
