'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-05 15:56:52
LastEditors: Shijie Cong
LastEditTime: 2024-01-09 13:56:15
'''
from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'tensorflow>=2.0.0',
    'absl-py',
    'matplotlib',
    'scikit-learn',
    'pandas',
    'seaborn',
    'tqdm',
    'munch',
    'pyyaml',
    'setuptools',
    'tensorboard',
    'torch>=2.1.0'
]

setup(
    name='DeepBSDE',
    version='0.0.1',
    install_requires=install_requires,
    description='DeepBSDE',
    author='Shijie Cong',
)