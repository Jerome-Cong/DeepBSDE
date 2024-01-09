'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-08 14:43:42
LastEditors: Shijie Cong
LastEditTime: 2024-01-09 21:55:16
'''
import logging
import time
import numpy as np
import torch
import torch.nn as nn

DELTA_CLIP = 50.0


class FeedForwardSubNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = nn.ModuleList()
        for _ in range(len(num_hiddens) + 2):
            self.bn_layers.append(nn.BatchNorm1d(dim, eps=1e-6, momentum=0.01))
        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(nn.Linear(dim, num_hiddens[0]))
        for i in range(len(num_hiddens) - 1):
            self.dense_layers.append(nn.Linear(num_hiddens[i], num_hiddens[i + 1]))
        self.dense_layers.append(nn.Linear(num_hiddens[-1], dim))
        
    def forward(self, x):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x)
            x = x.relu()
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        
        return x
    
class NonsharedModel(nn.Module):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.y_init = torch.empty((1), dtype=torch.float32)
        self.y_init.uniform_(self.net_config.y_init_range[0], self.net_config.y_init_range[1])
        self.z_init = torch.empty((1, self.eqn_config.dim), dtype=torch.float32)
        self.z_init.uniform_(-.1, .1)
        
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval - 1)]
        
    def forward(self, inputs):
        dw, x = inputs
        time_stamp = torch.arange(self.eqn_config.num_time_interval) * self.bsde.delta_t
        
        
        
class BSDESolver(object):
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        
        self.model = NonsharedModel(config, bsde)
        self.y_init = self.model.y_init
        