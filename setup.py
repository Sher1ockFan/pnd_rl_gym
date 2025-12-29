from setuptools import find_packages
from distutils.core import setup

setup(name='pnd_rl_gym',
      version='1.0.0',
      author='pndbotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@pndbotics.com',
      description='Template RL environments for PND Robots',
      install_requires=['matplotlib', 'numpy==1.20', 'tensorboard', 'mujoco==3.2.3', 'pyyaml', 'wandb'])
