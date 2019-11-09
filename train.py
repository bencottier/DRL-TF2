#!/usr/bin/env python
"""
main.py

Main program.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ddpg, td3
from utils import setup_logger_kwargs
import gym
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str,
            help='RL algorithm name, e.g. DDPG')
    parser.add_argument('env', type=str,
            help='environment name (from OpenAI Gym)')
    parser.add_argument('--hid', type=int, default=300,
            help='number of hidden units per hidden layer')
    parser.add_argument('--l', type=int, default=1,
            help='number of hidden layers')
    parser.add_argument('--discount', type=float, default=0.99,
            help='discount rate (denoted gamma in RL theory)')
    parser.add_argument('--seed', '-s', type=int, default=0,
            help='random seed')
    parser.add_argument('--epochs', type=int, default=50,
            help='number of epochs to train')
    parser.add_argument('--exp_name', type=str, default='unnamed',
            help='name for this experiment, used for the logging folder')
    args = parser.parse_args()

    algos = dict(ddpg=ddpg.run, td3=td3.run)

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=False)
    algos[args.algo.lower()](lambda : gym.make(args.env), logger_kwargs, args)
