'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-08 14:43:16
LastEditors: Shijie Cong
LastEditTime: 2024-01-16 10:34:58
'''
import numpy as np
import torch
import torch.nn as nn


class Equation(object):
    """Base class for defining PDE related function with torch."""
    
    def __init__(self, eqn_config) -> None:
        """
        :param dim: dimension of state variable
        """
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = float(self.total_time / self.num_time_interval)
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        
    def sample(self, num_sample):
        """Forward sample function for SDE."""
        raise NotImplementedError
    
    def f_torch(self, t, x, y, z):
        """Generator function in PDE."""
        raise NotImplementedError
    
    def g_torch(self, t, x):
        """Terminal condition of PDE."""
        raise NotImplementedError
    
    
class HJBLQ(Equation):
    """HJB equation for linear-quadratic case."""
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.)
        self.lambd = 1.
        
    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        
        return torch.tensor(dw_sample, dtype=torch.float32), torch.tensor(x_sample, dtype=torch.float32)
    
    def f_torch(self, t, x, y, z):
        return -self.lambd * torch.sum(torch.square(z), 1, keepdim=True) / 2
    
    def g_torch(self, t, x):
        return torch.log((1 + torch.sum(torch.square(x), 1, keepdim=True)) / 2)