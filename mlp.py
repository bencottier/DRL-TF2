#!/usr/bin/env python
"""
mlp.py

Multi-layer perceptron model.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
import numpy as np


class MLP(Model):

    def __init__(self, hidden_sizes, activation, output_activation=None, input_shape=None):
        super(MLP, self).__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.make_layers()
        if input_shape is not None:
            super(MLP, self).build(input_shape)

    def make_layers(self):
        for i, hidden_size in enumerate(self.hidden_sizes):
            act = self.activation if i < len(self.hidden_sizes) - 1 else self.output_activation
            layer = Dense(hidden_size, activation=act)
            setattr(self, f'dense{i}', layer)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    net = MLP([300, 400], 'relu', 'tanh')
    x = np.random.normal(size=(4, 17))
    print(x)
    print(net(x))
