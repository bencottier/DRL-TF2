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
            input_shape=None, **kwargs):
        super(RLEstimator, self).__init__()

    def polyak_update(self, other, p):
        self.set_weights(p * self.get_weights() + (1 - p) * other.get_weights())


class Actor(RLEstimator):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            action_space=None, input_shape=None, **kwargs):
        super(Actor, self).__init__()
        self.action_space = action_space
        act_dim = self.action_space.shape[0]
        self.act_limit = self.action_space.high[0]
        self.model = arch(list(hidden_sizes) + [act_dim], activation, 'tanh', input_shape)

    def call(self, x):
        return self.act_limit * self.model(x)

    def loss(self, q_pi):
        return -tf.reduce_mean(q_pi)


class Critic(RLEstimator):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            discount=0.99, input_shape=None, **kwargs):
        super(Critic, self).__init__()
        self.discount = discount
        self.model = arch(list(hidden_sizes) + [1], activation, None, input_shape)

    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)

    def loss(self, q, backup):
        return tf.reduce_mean((q - backup)**2)

    def bellman_backup(self, reward, done, qvalue):
        return tf.stop_gradient(reward + self.discount * (1 - done) * qvalue)
