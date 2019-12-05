#!/usr/bin/env python
"""
pretrain.py

Model pretraining to enhance learning.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from replay_buffer import ReplayBuffer
from logger import EpochLogger
from autoencoder import ConvolutionalAutoencoder
from utils import scale_float, scale_uint8
import tensorflow as tf
import numpy as np
import PIL
import time
import os


def generate_state_dataset(env_name, size=int(1e5), max_ep_len=1000):
    pass


def train_state_encoding(env_name, model_kwargs=dict(), seed=0, 
    steps_per_epoch=5000, epochs=100, lr=1e-3, batch_size=100, 
    logger_kwargs=dict(), save_freq=1):
    """

    """
    # Set up logging
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set random seed for relevant modules
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Create environment
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialise model used for encoding
    autoencoder = ConvolutionalAutoencoder(**model_kwargs)

    # Initialise replay buffer for storing and getting batches of transitions
    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=replay_size)

    # Set up model checkpointing so we can resume training or test separately
    checkpoint_dir = os.path.join(logger.output_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    model_dict = {f'state_autoencoder{i}': autoencoder}
    checkpoint = tf.train.Checkpoint(**model_dict)

    # @tf.function
    def train_step(batch):
        with tf.GradientTape(persistent=True) as tape:
            o_est = autoencoder(batch['obs1'], training=True)
            loss = tf.keras.losses.mean_squared_error(batch['obs1'], o_est)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        autoencoder.optimizer.apply_gradients(
            zip(gradients, autoencoder.trainable_variables))
        return loss

    


if __name__ == '__main__':
    import gym

    autoencoder = ConvolutionalAutoencoder([64, 64, 64, 1], 3)

    env_name = 'Hopper-v2'
    # env_name = 'Bowling-v0'
    env = gym.make(env_name)
    o = env.reset()
    im_frame = env.render(mode='rgb_array')

    print(im_frame.shape)

    import matplotlib.pyplot as plt

    im_pillow = PIL.Image.fromarray(im_frame)
    im_resize = im_pillow.resize((160, 160), resample=PIL.Image.BILINEAR)
    im = np.array(im_resize).astype(np.float32)
    im = scale_float(im)

    encoded_state = autoencoder(im[np.newaxis, ...], training=False)
    print(encoded_state)

    im_out = scale_uint8(autoencoder(im[np.newaxis, ...], training=True).numpy())

    plt.figure()
    plt.imshow(im_resize)
    plt.show()
    plt.figure()
    plt.imshow(im_out[0])
    plt.show()
