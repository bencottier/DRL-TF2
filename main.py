#!/usr/bin/env python
"""
main.py

Main program.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ddpg import ddpg
from utils import setup_logger_kwargs
import gym
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--arch', type=str, default='mlp')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), ac_arch=args.arch,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         discount=args.discount, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
