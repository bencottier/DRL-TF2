#!/usr/bin/env python
"""
model_visual.py

Visualise trained models.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from learn.pretrain import ObsObsLearner, StateObsLearner
from util.utils import setup_logger_kwargs, scale_uint8
import matplotlib.pyplot as plt
from matplotlib import animation
import gym
import PIL
import numpy as np


OBS_MEAN = 0.06712762
OBS_STD = 0.5819684


def compare_visual(env_name, model, num_samples=50, interval=10,
    im_size=(128, 128), max_ep_len=1000):
    
    env = gym.make(env_name)
    sample = 0
    while sample < num_samples:
        o, d, ep_len = env.reset(), False, 0

        base_frame = env.render(mode='rgb_array')
        base_frame_resized = PIL.Image.fromarray(base_frame).\
            resize(im_size, resample=PIL.Image.BILINEAR)
        base_frame_resized = np.array(base_frame)

        while not (d or (ep_len == max_ep_len)):
            if ep_len % interval == 0:
                true_frame = env.render(mode='rgb_array')
                true_frame = np.maximum(true_frame - base_frame, 0)
                true_frame = PIL.Image.fromarray(true_frame).\
                    resize(im_size, resample=PIL.Image.BILINEAR)
                true_frame = np.array(true_frame)

                # Predict the frame from the low-dimensional observation
                o_norm = (o - OBS_MEAN) / OBS_STD
                pred = model(o_norm.reshape(1, *o.shape)).numpy()
                pred = pred.reshape(true_frame.shape)
                pred = (pred*255).astype(np.uint8)
                # Prediction is the difference from a base frame
                # Add back for better (not perfect) comparison
                # pred_frame = base_frame + pred
                pred_frame = pred

                # Compare frames visually
                plt.figure(figsize=(14, 12))
                plt.subplot(121)
                plt.imshow(true_frame)
                plt.subplot(122)
                plt.imshow(pred_frame)
                plt.show()

                sample += 1
                if sample >= num_samples:
                    break

            # Step the environment
            a = env.action_space.sample()  # random action
            o, _, d, _ = env.step(a)
            ep_len += 1


def main():
    exp_name = 'decode-state-diff'
    seed = 20200103
    env_name = 'LunarLanderContinuous-v2'
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    learner = StateObsLearner(env_name, channels=3,
        model_kwargs=dict(lr=1e-3, hidden_sizes=[64, 64, 64, 64, 3], 
        kernel_size=4), logger_kwargs=logger_kwargs, seed=seed)
    learner.load_model(checkpoint_number=None)

    compare_visual(env_name, learner.model)


if __name__ == '__main__':
    main()
