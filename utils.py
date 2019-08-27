#!/usr/bin/env python
"""
utils.py

Utility functions.

author: Ben Cottier (git: bencottier)
"""
import time
import os.path as osp
import json


FORCE_DATESTAMP = True
DEFAULT_DATA_DIR = './out'


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
