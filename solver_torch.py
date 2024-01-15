'''
Description: 
Version: 1.0
Autor: Shijie Cong
Date: 2024-01-08 14:43:42
LastEditors: Shijie Cong
LastEditTime: 2024-01-15 16:53:33
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
        self.y_init.requires_grad = True
        self.z_init = torch.empty((1, self.eqn_config.dim), dtype=torch.float32)
        self.z_init.uniform_(-.1, .1)
        self.z_init.requires_grad = True
        
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval - 1)]
        
    def forward(self, inputs):
        dw, x = inputs
        time_stamp = torch.arange(self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = torch.ones([dw.size()[0], 1], dtype=torch.float32)
        y = all_one_vec * self.y_init
        z = torch.matmul(all_one_vec, self.z_init)
        
        for t in range(0, self.bsde.num_time_interval - 1):
            y = y - self.bsde.delta_t * (self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z)) + \
                torch.sum(z * dw[:, :, t], 1, keepdim=True)
            z = self.subnet[t](x[:, :, t + 1]) / self.bsde.dim  # ??? why divide by dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_torch(time_stamp[-1], x[:, :, -2], y, z) + \
            torch.sum(z * dw[:, :, -1], 1, keepdim=True)
            
        return y
        
        
class BSDESolver(object):
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        
        self.model = NonsharedModel(config, bsde)
        self.y_init = self.model.y_init
        
        init_lr = self.net_config.lr_values[0]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.net_config.lr_boundaries,
                                                              verbose=self.net_config.verbose)
    
    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)
        
        # sgd iteration
        for step in range(self.net_config.num_iterations+1):
            if step % self.net_config.logging_frequency == 0:
                
                loss = self.loss_fn(valid_data)
                y_init = self.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    loss: %.4e,    valid_error: %.4e,   elapsed_time: %3u" %
                                 (step, loss, y_init, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config.batch_size))
        
        return np.array(training_history)
                
    # raw loss fn            
    def loss_fn(self, inputs):
        dw, x = inputs
        y_terminal = self.bsde.g_torch(self.bsde.total_time, x[:, :, -1])
        y_pred = self.model(inputs)
        delta = y_pred - y_terminal
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, torch.square(delta),
                                      2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))
        # loss = torch.mean(torch.square(y_pred - y_terminal))
        
        
        return loss
    
    # def grad(self, inputs):
    #     dw, x = inputs
    #     y_terminal = self.bsde.g_torch(self.bsde.total_time, x[:, :, -1])
    #     y_pred = self.model(inputs)
    #     loss = torch.mean(torch.square(y_pred - y_terminal))
    #     loss.backward()
        
    #     return loss
    
    def train_step(self, inputs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(inputs)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss