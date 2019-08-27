#!/usr/bin/env python
"""
actor_critic.py

Actor and critic models.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np


class RLEstimator(Model):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            input_shape=None, lr=None, **kwargs):
        super(RLEstimator, self).__init__()
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def polyak_update(self, other, p):
        for (ws, wo) in zip(self.trainable_variables, other.trainable_variables):
            ws.assign(p * ws + (1 - p) * wo)


class Actor(RLEstimator):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            action_space=None, input_shape=None, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.action_space = action_space
        act_dim = self.action_space.shape[0]
        self.act_limit = self.action_space.high[0]
        self.model = arch(list(hidden_sizes) + [act_dim], activation, 'tanh', input_shape)

    def call(self, x):
        return self.act_limit * self.model(x)

    def loss(self, q_pi):
        return -tf.reduce_mean(q_pi)

    @tf.function
    def train_step(self, batch, critic):
        with tf.GradientTape() as tape:
            pi = self(batch['obs1'])
            q_pi = critic(batch['obs1'], pi)
            loss = self.loss(q_pi)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class Critic(RLEstimator):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), 
            activation='relu', input_shape=None, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.model = arch(list(hidden_sizes) + [1], activation, None, input_shape)

    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)

    def loss(self, q, backup):
        return tf.reduce_mean((q - backup)**2)

    @staticmethod
    def bellman_backup(discount, reward, done, qvalue):
        return tf.stop_gradient(reward + discount * (1 - done) * qvalue)

    @tf.function
    def train_step(self, batch, critic_target, actor_target, discount):
        with tf.GradientTape() as tape:
            q = self(batch['obs1'], batch['acts'])
            q_pi_targ = critic_target(batch['obs2'], actor_target(batch['obs2']))
            backup = critic_target.bellman_backup(discount, batch['rwds'], batch['done'], q_pi_targ)
            loss = self.loss(q, backup)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, q
