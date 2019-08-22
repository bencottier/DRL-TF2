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


class Actor(Model):

    def __init__(self, obs_dim, act_dim, action_space, arch=MLP, 
            hidden_sizes=(400, 300), activation='relu', **kwargs):
        super(Actor, self).__init__()
        self.model = arch((obs_dim,), list(hidden_sizes) + [act_dim], 
                activation, output_activation='tanh')
        self.action_space = action_space
        self.act_limit = self.action_space.high[0]

    def call(self, x):
        return self.act_limit * self.model(x)


class Critic(Model):

    def __init__(self, obs_dim, act_dim, arch=MLP, 
            hidden_sizes=(400, 300), activation='relu', **kwargs):
        super(Critic, self).__init__()
        self.model = arch((obs_dim + act_dim,), list(hidden_sizes) + [1], activation, None)

    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)

