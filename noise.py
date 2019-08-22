#!/usr/bin/env python
"""
noise.py

Classes to generate noise, or more technically, stochastic processes.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


class StochasticProcess:
    def __init__(self, loc=0., shape=None, x0=None, dt=1.0):
        self.loc = loc
        self.shape = shape
        self.x0 = np.zeros(self.shape, dtype=np.float32) if x0 is None else x0
        self.dt = dt
        self.reset()

    def next_value(self):
        return 0.0

    def sample(self, update=True):
        x = self.next_value()
        if update:
            self.x = x
        return x

    def reset(self):
        self.x = self.x0


class StochasticProcessWithScale(StochasticProcess):
    def __init__(self, scale=1.0, *args, **kwargs):
        super(StochasticProcessWithScale, self).__init__(*args, **kwargs)
        self.scale = scale


class NormalProcess(StochasticProcessWithScale):
    """
    Samples from a normal distribution independently at each step.
    """
    def next_value(self):
        return np.random.normal(self.loc, self.scale, self.shape)


class OUProcess(StochasticProcessWithScale):
    def __init__(self, theta=0.15, sigma=0.2, *args, **kwargs):
        super(OUProcess, self).__init__(scale=sigma, *args, **kwargs)
        self.theta = theta
        self.sqrtdt = np.sqrt(self.dt)

    def next_value(self):
        return self.x + self.theta * (self.loc - self.x) * self.dt + \
                self.scale * self.sqrtdt * np.random.normal(size=self.shape)


if __name__ == '__main__':
    n = 1000
    proc = OUProcess(dt=1/n)
    x = np.arange(n)
    y = np.sin(16.*np.pi*x/n)
    noise = np.array([proc.sample() for _ in range(n)])

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x, y + 0.1*np.random.randn(n))
    plt.plot(x, y + noise)
    plt.legend(['$\sin(k\pi x/n)$', 'with normal noise', 'with OU noise'])
    plt.show()
