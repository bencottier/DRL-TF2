#!/usr/bin/env python
"""
utils.py

Utility functions.

author: Ben Cottier (git: bencottier)
"""

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])
