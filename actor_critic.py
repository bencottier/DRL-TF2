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

    @tf.function
    def call(self, x):
        return self.act_limit * self.model(x)


class Critic(RLEstimator):
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), 
            activation='relu', input_shape=None, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.model = arch(list(hidden_sizes) + [1], activation, None, input_shape)

    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)
