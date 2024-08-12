import logging
import time
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import datetime

DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, save_path=None):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.device = bsde.device

        self.save_path = save_path
        self.model = NonsharedModel(config, bsde, device=self.device)
        self.y_init = None
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values)
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(self.net_config.lr_values[0],
        #                                                                 100, t_mul=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        train_stamp = []
        train_losses = []
        train_lrs = []
        valid_data = self.bsde.sample(self.net_config.valid_size, seed=3407, lqr=False)
        lowest_val_loss = np.inf

        # begin sgd iteration
        for step in range(self.net_config.num_iterations+1):
            train_data = self.bsde.sample(self.net_config.batch_size, lqr=False)
            lr = self.lr_schedule(step)
            train_lrs.append(lr)
            train_loss = self.train_step(train_data)
            train_losses.append(train_loss)
            train_stamp.append(step)
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.model.y_init.numpy()[0].mean()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
                if loss < lowest_val_loss:
                    lowest_val_loss = loss
                    # self.model.save(os.path.join(self.save_path,f'Epoch{step}'+'_full_model'), format='tf')
        # self.model.save(os.path.join(self.save_path,f'Epoch{step}'+'_full_model'))
        metrics = {'train_stamp': train_stamp, 'train_losses': train_losses, 'train_lr': train_lrs}
        # raise ValueError('Training works fine')
        self.eval_fn(metrics, np.array(training_history))
        # np.save(os.path.join(self.save_path,'eval_results.npy'), eval_results)
        return np.array(training_history)

    def loss_fn(self, inputs, training):
        dw, x, _, _ = inputs
        y_terminal = self.model(inputs, training)
        delta = y_terminal - self.bsde.g_tf(x[:, :, -1])
        # use linear approximation outside the clipped range
        # loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                    #    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss = tf.reduce_mean(tf.square(delta))

        return loss
    
    def eval_fn(self, metrics, train_history):
        # zs = np.zeros((4096,1, self.bsde.num_time_interval))
        # x_range = [-10,10]
        # xs = np.random.uniform(x_range[0], x_range[1], size=[4096,1])
        # v0 = self.model.value_net(xs).numpy()
        # for i in range(self.bsde.num_time_interval):
        #     zs[:,:,i] = self.model.subnet[i](xs).numpy()
        
        # result = {
        #     'grad': zs,
        #     'v0': v0,
        #     'xs': xs
        # }
        horizon = self.bsde.num_time_interval
        x1s = np.random.uniform(size=[4096, 2, horizon])*20-5.
        x1s[:,1,:] = 0.
        x2s = np.random.uniform(size=[4096, 2, horizon])*20-10.
        x2s[:,0,:] = 0.
        z1s = np.zeros([4096,2,horizon])
        z2s = np.zeros([4096,2,horizon])
        v01 = self.model.value_net(x1s[:,:,0]).numpy()
        v02 = self.model.value_net(x2s[:,:,0]).numpy()
        for i in range(horizon):
            z1s[:,:,i] = self.model.subnet[i](x1s[:,:,i]).numpy()
            z2s[:,:,i] = self.model.subnet[i](x2s[:,:,i]).numpy()
            
        timestamp=np.arange(horizon)

        exp_info = {
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Exp. Name': 'lqr_2d',
            # 'Ref. Policy': r'Control law by solving Riccati equation and clipped by $[-10,10]$',
            'Ref. Policy': r'Uni. sampling in $[-10,10]$',
            'Horizon': f'T={self.bsde.total_time}, dt={self.bsde.delta_t}',
            'Init. state distribution': f's0 uni. in {self.bsde.x_range}, v=0, ckpt={self.bsde.ckpt}',
            'Learning Rate': self.net_config.lr_values[0],
            'NN Archi.': '3 hidden layers with 32 neurons each',
            # 'Activation': co.net_config.activation,
            'Batch Size': self.net_config.batch_size,
            'Validation Size': self.net_config.valid_size,
            
        }
        n_row = 27
        n_col = 3
        assert n_row * n_col >= horizon + 1
        fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(30,60))
        t = 0
        for i in range(n_row):
            for j in range(n_col):
                if t<horizon:
                    axs[i,j].scatter(x1s[:,0,t], z1s[:,0,t])
                    axs[i,j].set_xlabel('Position')
                    axs[i,j].set_ylabel('Approx. gradient')
                    axs[i,j].set_title(f'Approx. gradient vs position at time stamp {t}')
                    t += 1
                else:
                    pass

        axs[-1,-1].scatter(x1s[:,0,0], v01)
        axs[-1,-1].set_xlabel('Position')
        axs[-1,-1].set_ylabel('Value')
        axs[-1,-1].set_title(f'Value vs position at time stamp 0')


        fig.subplots_adjust(top=0.6)
        keys = list(exp_info.keys())
        vals = list(exp_info.values())

        n_ins = 5
        n_NN = 2
        n_solver = 2
        ins_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins)])
        NN_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins, n_ins+n_NN)])
        solver_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins+n_NN, n_ins+n_NN+n_solver)])

        fig.text(0.05, 0.82, ins_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig.text(0.45, 0.82, NN_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig.text(0.85, 0.82, solver_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.tight_layout(rect=[0, 0, 1, 0.8])
        save_path = os.path.join(self.save_path, 'gradient_pos.png')
        plt.savefig(save_path)
        plt.clf()
        
        fig2, ax2s = plt.subplots(nrows=n_row, ncols=n_col, figsize=(30,60))
        t = 0
        for i in range(n_row):
            for j in range(n_col):
                if t<horizon:
                    ax2s[i,j].scatter(x2s[:,1,t], z2s[:,1,t])
                    ax2s[i,j].set_xlabel('Velocity')
                    ax2s[i,j].set_ylabel('Approx. gradient')
                    ax2s[i,j].set_title(f'Approx. gradient vs velocity at time stamp {t}')
                    t += 1
                else:
                    pass

        ax2s[-1,-1].scatter(x2s[:,1,0], v02)
        ax2s[-1,-1].set_xlabel('Velocity')
        ax2s[-1,-1].set_ylabel('Value')
        ax2s[-1,-1].set_title(f'Value vs position at time stamp 0')


        fig2.subplots_adjust(top=0.6)
        keys = list(exp_info.keys())
        vals = list(exp_info.values())

        n_ins = 5
        n_NN = 2
        n_solver = 2
        ins_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins)])
        NN_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins, n_ins+n_NN)])
        solver_text = '\n'.join([f'{keys[i]}: {vals[i]}' for i in range(n_ins+n_NN, n_ins+n_NN+n_solver)])

        fig2.text(0.05, 0.82, ins_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig2.text(0.45, 0.82, NN_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        fig2.text(0.85, 0.82, solver_text, ha='left', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.tight_layout(rect=[0, 0, 1, 0.8])
        save_path = os.path.join(self.save_path, 'gradient_vel.png')
        plt.savefig(save_path)
        plt.clf()
        
        
        fig1, ax1 = plt.subplots(2, 1, figsize=(8, 12))
        ax1[0].set_xlabel('Iterations')
        ax1[0].set_ylabel('Loss')
        ax1[0].semilogy(metrics['train_stamp'], metrics['train_losses'], label='Training loss')
        ax1[0].semilogy(train_history[:, 0], train_history[:, 1], label='Validation loss')
        ax1[0].legend()
        ax1[1].set_xlabel('Iterations')
        ax1[1].set_ylabel('Learning rate')
        ax1[1].semilogy(metrics['train_stamp'], metrics['train_lr'], label='Learning rate')
        plt.tight_layout()
        save_path = os.path.join(self.save_path, 'log_plot.png')
        plt.savefig(save_path)
        # return result

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad, loss

    @tf.function
    def train_step(self, train_data):
        grad, loss = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return loss


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde, device):
        super(NonsharedModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
        self.device = device

        self.value_net = FeedForwardNet(config)
        self.subnet = [FeedForwardSubNet(config) for _ in range(self.bsde.num_time_interval)]

    def call(self, inputs, training):
        dw, x, u, h = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t

        y = self.value_net(x[:, :, 0], training)
        self.y_init = y

        for t in range(self.bsde.num_time_interval):
            z = self.subnet[t](x[:, :, t], training) / self.bsde.dim
            dy = self.bsde.delta_t * self.bsde.f_tf(x[:,:,t], u[:,:,t], h[:,:,t], z) + \
                self.bsde.sigma * tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            y = y + dy

        return y


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__()
        dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        # x = self.bn_layers[-1](x, training)
        return x
    
    # def get_config(self):
    #     config = super(FeedForwardSubNet, self).get_config()
    #     config.update({
    #         'eqn_config': self.eqn_config,
    #         'net_config': self.net_config
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class FeedForwardNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardNet, self).__init__()
        dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(1, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        # x = self.bn_layers[-1](x, training)
        return x
    
    # def get_config(self):
    #     config = super(FeedForwardNet, self).get_config()
    #     config.update({
    #         'eqn_config': self.eqn_config,
    #         'net_config': self.net_config
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)