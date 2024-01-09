'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-08 14:43:31
LastEditors: Shijie Cong
LastEditTime: 2024-01-09 15:49:59
'''
import json
import os
import munch
import logging

from absl import flags
from absl import app
from absl import logging as absl_logging
import numpy as np
import torch

import equation_torch as eqn
import solver_torch as solver


flags.DEFINE_string('config_path', 'configs/hjb_lq_d100.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', 'test',
                    """The name of numerical experiments, prefix for logging""")
FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array