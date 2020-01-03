#!/usr/bin/env python
"""
autoencoder.py

Convolutional autoencoder model.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import math


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, 
        padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, 
        padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


class ConvDecoder(tf.keras.Model):

    def __init__(self, hidden_sizes=(64, 64, 3), input_shape=(64, 1),
        kernel_size=3, lr=None):
        super(ConvDecoder, self).__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.kernel_size = kernel_size
        self._input_shape = input_shape
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)
        self.make_layers()

    def make_layers(self):
        self.up_stack = list()

        dense_unit = self.hidden_sizes[0]
        dense_sqrt = int(math.sqrt(dense_unit))
        assert dense_sqrt**2 == dense_unit, \
            'dense units must be a square number'
        self.up_stack.append(tf.keras.Sequential([
            tf.keras.layers.Dense(dense_unit),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((dense_sqrt, dense_sqrt, 1))]))

        # Build upsampling convolution stack
        drop = min(len(self.hidden_sizes), 3)
        self.up_stack += [upsample(f, self.kernel_size, apply_dropout=(i<drop))
            for i, f in enumerate(self.hidden_sizes[1:-1])]
        # Final layer with output channels
        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(self.hidden_sizes[-1], 
            self.kernel_size, strides=2, padding='same', 
            kernel_initializer=initializer, activation='tanh')
        
    def call(self, x, training=None):
        for up in self.up_stack: x = up(x)
        return self.last(x)


class ConvAutoencoder(tf.keras.Model):

    def __init__(self, hidden_sizes=(64, 1), input_shape=(64, 64, 3), 
        latent_dim=None, kernel_size=3, lr=None):
        super(ConvAutoencoder, self).__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.kernel_size = kernel_size
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.output_channels = input_shape[2]
        self.latent_dim = latent_dim
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)
        self.make_layers()

    def make_layers(self):
        """
        https://www.tensorflow.org/tutorials/generative/pix2pix
        """
        # Encoder layers
        self.down_stack = [downsample(f, self.kernel_size, apply_batchnorm=(i!=0)) 
                for i, f in enumerate(self.hidden_sizes)]
        self.up_stack = list()
        # Optional dense dimensionality conversion
        if self.latent_dim is not None:
            restore_shape = (int(self.height/(2**len(self.hidden_sizes))), 
                int(self.width/(2**len(self.hidden_sizes))), 1)
            restore_units = restore_shape[0] * restore_shape[1]
            # Reduce to latent dimension
            self.down_stack.append(tf.keras.layers.Flatten())
            self.down_stack.append(tf.keras.layers.Dense(
                self.latent_dim, activation='relu'))
            # Restore to previous dimension
            self.up_stack.append(tf.keras.layers.Dense(
                restore_units, activation='relu'))
            self.up_stack.append(tf.keras.layers.Reshape(restore_shape))
        # Decoder layers
        drop = min(len(self.hidden_sizes), 3)
        self.up_stack += [upsample(f, self.kernel_size, apply_dropout=(i<drop))
                for i, f in enumerate(self.hidden_sizes[-2::-1])]
        # Restore input
        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(self.output_channels, self.kernel_size,
                strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    def encode(self, x):
        for down in self.down_stack: x = down(x)
        return (tf.keras.layers.Flatten())(x)
    
    def call(self, x, training=None):
        for down in self.down_stack: x = down(x)
        for up in self.up_stack: x = up(x)
        return self.last(x)


if __name__ == '__main__':
    # input_shape = (160, 160, 3)
    # autoencoder = ConvAutoencoder(hidden_sizes=(64, 64, 64, 1),
    #     input_shape=input_shape, latent_dim=None, kernel_size=4)
    # # outputs = autoencoder(np.random.randn(2, 32, 32, 3).astype(np.float32))
    # inputs = tf.keras.layers.Input(shape=input_shape)
    # outputs = autoencoder.call(inputs, training=True)
    # autoencoder_fn = tf.keras.Model(inputs=inputs, outputs=outputs)
    # tf.keras.utils.plot_model(autoencoder_fn, 
    #     to_file='../out/visual/autoencoder_example.png', 
    #     show_shapes=True, dpi=64)

    input_shape = (11,)
    decoder = ConvDecoder(hidden_sizes=(8, 16, 32, 64, 3), 
        input_shape=input_shape, kernel_size=4)
    inputs = tf.keras.layers.Input(shape=input_shape)
    outputs = decoder.call(inputs, training=True)
    decoder_fn = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(decoder_fn, 
        to_file='../out/visual/decoder_example.png',
        show_shapes=True, dpi=64)
