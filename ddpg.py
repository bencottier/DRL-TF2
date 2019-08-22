#!/usr/bin/env python
"""
ddpg.py

Main script for the deep deterministic policy gradient algorithm.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
    

def ddpg(env_fn, ac_arch=MLP, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
         act_noise=0.1, max_ep_len=1000, save_freq=1):
    # TODO: logging
    # Set random seed for relevant modules
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # Create environment
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    # Randomly initialise critic and actor networks
    critic = Critic(input_shape=(batch_size, obs_dim + act_dim), **ac_kwargs)
    actor = Actor(input_shape=(batch_size, obs_dim), **ac_kwargs)
    # Initialise target networks
    critic_target = Critic(input_shape=(batch_size, obs_dim + act_dim), **ac_kwargs)
    actor_target = Actor(input_shape=(batch_size, obs_dim), **ac_kwargs)
    critic_target.set_weights(critic.get_weights())
    actor_target.set_weights(actor.get_weights())
    # Initialise replay buffer
    replay_buffer = ReplayBuffer(size=replay_size)
