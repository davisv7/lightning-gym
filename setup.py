from setuptools import setup, find_packages

setup(
    name='lightning_gym',
    version='1.0',
    description='Gym Environment for the Bitcoin Lightning Network',
    author='Vincent',
    author_email='vmda225@uky.edu',
    packages=find_packages(include=['lightning_gym']),  # same as name
    install_requires=[
        "gym",
        "numpy",
        "bidict",
        "dgl",
        "torch",
        "matplotlib",
        "networkx",
        "python-igraph"
    ],  # external packages as dependencies
)
