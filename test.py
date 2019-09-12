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


def test_policy(output_dir, env_name, episodes=10):
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
    env_fn = lambda : gym.make(env_name)
    env, test_env = env_fn(), env_fn()
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
    # checkpoint.restore(os.path.join(checkpoint_dir, 'ckpt-1')).expect_partial()
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    # Run policy for specified number of episodes, recording return
    ep_rets = np.zeros(episodes)
    for i in range(episodes):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not(d or (ep_len == exp_data['max_ep_len'])):
            test_env.render()
            o, r, d, _ = test_env.step(actor(o.reshape(1, -1)))
            ep_ret += r
            ep_len += 1
        ep_rets[i] = ep_ret
        print(f'Episode {i}: return={ep_ret:.0f} length={ep_len}')
    # Summary stats
    print(f'avg={ep_rets.mean():.0f} std={ep_rets.std():.0f} ' \
            f'min={ep_rets.min():.0f} max={ep_rets.max():.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./out/ddpg-benchmark-cheetah-s0')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    test_policy(args.output_dir, args.env_name, args.episodes)
