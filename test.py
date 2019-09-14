#!/usr/bin/env/python
"""
test.py

Test a policy saved from a checkpoint.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from actor_critic import Actor, Critic
import gym
import tensorflow as tf
import numpy as np
import json
import argparse
import os


def test_policy(output_dir, env_name, episodes, checkpoint_number):
    """
    Run a learned policy with visualisation in the environment.

    Args:
        output_dir: str. Directory containing a JSON file named 
            'config.json' with experiment metadata, and a subfolder
            named 'training_checkpoints' containing model checkpoints.
        env_name: str. Name of the environment to run the policy in.
        episodes: int. Number of episodes to run the policy for.
    """
    # Load experimental metadata from file
    with open(os.path.join(output_dir, 'config.json'), 'r') as f:
        exp_data = json.load(f)

    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Share information about action space with policy architecture
    ac_kwargs = dict(hidden_sizes=exp_data['ac_kwargs']['hidden_sizes'])
    ac_kwargs['action_space'] = env.action_space

    # Randomly initialise critic and actor networks
    critic = Critic(input_shape=(exp_data['batch_size'], obs_dim + act_dim), **ac_kwargs)
    actor = Actor(input_shape=(exp_data['batch_size'], obs_dim), **ac_kwargs)

    # Optimizers
    critic_optimizer = tf.keras.optimizers.Adam(exp_data['q_lr'])
    actor_optimizer = tf.keras.optimizers.Adam(exp_data['pi_lr'])

    checkpoint_dir = os.path.join(output_dir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(critic_optimizer=critic_optimizer,
                                    actor_optimizer=actor_optimizer,
                                    critic=critic,
                                    actor=actor)
    if checkpoint_number is not None:
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{checkpoint_number}')).expect_partial()
    else:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # Run policy for specified number of episodes, recording return
    ep_rets = np.zeros(episodes)
    for i in range(episodes):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        while not(d or (ep_len == exp_data['max_ep_len'])):
            env.render()
            o, r, d, _ = env.step(actor(o.reshape(1, -1)))
            if type(r) == np.ndarray:
                r = r[0]
            ep_ret += r
            ep_len += 1
        ep_rets[i] = ep_ret
        print(f'Episode {i}: return={ep_ret:.0f} length={ep_len}')
    # Summary stats
    print(f'avg={ep_rets.mean():.0f} std={ep_rets.std():.0f} ' \
            f'min={ep_rets.min():.0f} max={ep_rets.max():.0f}')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
            help='directory containing config.json and training_checkpoints folder')
    parser.add_argument('env', type=str,
            help='environment name, normally matches training')
    parser.add_argument('--episodes', type=int, default=10,
            help='number of episodes to run')
    parser.add_argument('--checkpoint', type=int, default=None,
            help='checkpoint to load models from (default latest)')
    args = parser.parse_args()

    test_policy(args.dir, args.env, args.episodes, args.checkpoint)
