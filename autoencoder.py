#!/usr/bin/env python
"""
autoencoder.py

Convolutional autoencoder model.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


class ConvolutionalAutoencoder(tf.keras.Model):

    def __init__(self, hidden_sizes=(64, 64, 64, 1), kernel_size=3, 
        output_channels=3, lr=None):
        super(ConvolutionalAutoencoder, self).__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        if lr is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr)
        self.make_layers()

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, 
                strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, 
                strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def make_layers(self):
        """
        https://www.tensorflow.org/tutorials/generative/pix2pix
        """
        # Encoder layers
        self.down_stack = [self.downsample(f, self.kernel_size, apply_batchnorm=(i!=0)) 
                for i, f in enumerate(self.hidden_sizes)]
        # Decoder layers
        drop = max(len(self.hidden_sizes), 3)
        self.up_stack = [self.upsample(f, self.kernel_size, apply_dropout=(i<drop))
                for i, f in enumerate(self.hidden_sizes[-2::-1])]
        # Back to input shape
        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(self.output_channels, self.kernel_size,
                strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    def call_fn(self, x, training=False):
        for down in self.down_stack: x = down(x)
        if not training: return (tf.keras.layers.Flatten())(x)
        for up in self.up_stack: x = up(x)
        return self.last(x)
    
    def call(self, x, training=False):
        for down in self.down_stack: x = down(x)
        if not training: return (tf.keras.layers.Flatten())(x)
        for up in self.up_stack: x = up(x)
        return self.last(x)


if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder([64, 64, 64, 1], 3)
    # outputs = autoencoder(np.random.randn(2, 32, 32, 3).astype(np.float32))
    inputs = tf.keras.layers.Input(shape=(160, 160, 3))
    outputs = autoencoder.call_fn(inputs, training=False)
    autoencoder_fn = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(autoencoder_fn, to_file='encoder_example.png', show_shapes=True, dpi=64)
