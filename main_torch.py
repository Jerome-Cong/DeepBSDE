'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-08 14:43:31
LastEditors: Shijie Cong
LastEditTime: 2024-01-15 19:20:42
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

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', 'configs/hjb_lq_d100.json',
                    'The path to load json file.')
flags.DEFINE_string('exp_name', 'test',
                    'The name of numerical experiments, prefix for logging')

FLAGS.log_dir = './logs'  # directory where to write event logs and output array

def main(argv):
    del argv
    
    with open(FLAGS.config_path) as f:
        config = json.load(f)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name)) for name in dir(config)
                       if not name.startswith('__')), outfile, indent=2)
        
    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')
    
    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    bsde_solver = solver.BSDESolver(config, bsde)
    training_history = bsde_solver.train()
    if bsde.y_init is not None:
        logging.info('Y0_true: %.4e' % (bsde.y_init))
        logging.info('relative error of Y0: %s', '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init))
    np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=',',
               header='step,loss_function,target_value,elapsed_time',
               comments='')
    
    
    
if __name__ == '__main__':
    app.run(main)