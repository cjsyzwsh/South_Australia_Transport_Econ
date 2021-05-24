# sw: not necessary at this point since it is about publishing the repo as a package.

from setuptools import find_packages, setup

setup(
    name='network analysis for transport and economy',
    packages=find_packages(),
    version='0.1.0',
    description='Analyze the relationship between economic growth and transport through network analysis',
    author='Shenhao Wang',
    license='MIT',
)
