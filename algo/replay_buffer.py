"""
replay_buffer.py

Replay buffer class.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


class ReplayBuffer:
    """
    A simple replay buffer to store transitions from an RL agent 
    interacting with an environment.
    """
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        """
        Initialize a ReplayBuffer.

        Args:
            obs_dim. int. Size of the observation vector.
            act_dim. int. Size of the action vector.
            size: int. Maximum size of the buffer.
        """
        self.idx = 0
        self.size = 0
        self.max_size = size
        self.obs1_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buffer = np.zeros((size, act_dim), dtype=np.float32)
        self.rwds_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rwd, next_obs, done):
        """
        Store a transition.

        If the buffer is at maximum size, this will overwrite the 
        oldest transition in the buffer.
        """
        self.obs1_buffer[self.idx] = obs
        self.acts_buffer[self.idx] = act
        self.rwds_buffer[self.idx] = rwd
        self.obs2_buffer[self.idx] = next_obs
        self.done_buffer[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions.

        Args:
            batch_size: int. Number of transitions in the batch.
        Returns:
            A dict containing the batch, with these key-value pairs:
                'obs1': first observation
                'obs2': second observation
                'acts': action
                'rwds': reward
                'done': terminal state indicator
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
                obs1=self.obs1_buffer[idxs], 
                obs2=self.obs2_buffer[idxs], 
                acts=self.acts_buffer[idxs], 
                rwds=self.rwds_buffer[idxs], 
                done=self.done_buffer[idxs])
