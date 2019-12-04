#!/usr/bin/env python
"""
pretrain.py

Model pretraining to enhance learning.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from autoencoder import ConvolutionalAutoencoder
import gym
import numpy as np
import PIL


if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder([64, 64, 64, 1], 3)

    env_name = 'Hopper-v2'
    # env_name = 'Bowling-v0'
    env = gym.make(env_name)
    o = env.reset()
    im_frame = env.render(mode='rgb_array')

    import matplotlib.pyplot as plt

    img = PIL.Image.fromarray(im_frame)
    img = img.resize((160, 160), resample=PIL.Image.BILINEAR)
    img = np.array(img).astype(np.float32)
    img = -1. + 2. * (img - img.min()) / (img.max() - img.min())

    img_out = autoencoder(img[np.newaxis, ...], training=True)

    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(img_out[0])
    plt.show()

    print(im_frame.shape)
