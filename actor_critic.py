#!/usr/bin/env python
"""
actor_critic.py

Actor and critic models.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
import tensorflow as tf
import numpy as np


class RLEstimator(tf.keras.Model):
    """
    Base class for gradient-based parametric estimators used in 
    reinforcement learning algorithms.
    """
    def __init__(self, lr=None, **kwargs):
        """
        Initialize an RLEstimator.

        Args:
            lr: float. Learning rate for this estimator's optimizer.
        """
        super(RLEstimator, self).__init__()
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def polyak_update(self, other, p):
        """
        Updates the variables of this estimator as a "polyak" average
        of the current values and the values of the corresponding 
        variables in `other`, weighted by `p`.
        """
        for (ws, wo) in zip(self.trainable_variables, other.trainable_variables):
            ws.assign(p * ws + (1 - p) * wo)


class Actor(RLEstimator):
    """
    Actor component of an actor-critic model. Equivalent to a 
    policy estimate function, and can be called as such.
    """
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), activation='relu', 
            action_space=None, input_shape=None, **kwargs):
        """
        Initialize an Actor estimator.

        Args:
            arch: keras.Model. A parametric model architecture. 
                Must implement the same interface as mlp.MLP.
            hidden_sizes: tuple of int. Number of hidden units for each 
                layer in the model architecture.
            activation: str. Name of the hidden activation function in 
                the model architecture.
            action_space: gym.Space. Action space of the environment.
            input_shape: int or tuple of int. Shape of the tensor input
                to the model.
        """
        super(Actor, self).__init__(**kwargs)
        self.action_space = action_space
        act_dim = self.action_space.shape[0]
        self.act_limit = self.action_space.high[0]
        self.model = arch(list(hidden_sizes) + [act_dim], activation, 'tanh', input_shape)

    @tf.function
    def call(self, x):
        return self.act_limit * self.model(x)


class Critic(RLEstimator):
    """
    Critic component of an actor-critic model. Equivalent to a 
    Q estimate function, and can be called as such.
    """
    def __init__(self, arch=MLP, hidden_sizes=(400, 300), 
            activation='relu', input_shape=None, **kwargs):
        """
        Initialize a Critic estimator.

        Args:
            arch: keras.Model. A parametric model architecture. 
                Must implement the same interface as mlp.MLP.
            hidden_sizes: tuple of int. Number of hidden units for each 
                layer in the model architecture.
            activation: str. Name of the hidden activation function in 
                the model architecture.
            input_shape: int or tuple of int. Shape of the tensor input
                to the model.
        """
        super(Critic, self).__init__(**kwargs)
        self.model = arch(list(hidden_sizes) + [1], activation, None, input_shape)

    @tf.function
    def call(self, x, a):
        return tf.squeeze(self.model(tf.concat([x, a], axis=1)), axis=1)
