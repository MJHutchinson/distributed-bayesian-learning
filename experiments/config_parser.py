from copy import deepcopy
from itertools import product

def deep_set(d,nested_key,value):
    keylist = nested_key.split('.')
    val = d
    for key in keylist[:-1]:
         val = val[key]
    val[keylist[-1]] = value
    return d

def parse_sweep_config(config):
    base_config = config['base_config']
    sweep_config = config['sweep_config']

    sweep_type = sweep_config['type']

    sweep_params = sweep_config['sweep']

    keys = sweep_params.keys()
    values = [sweep_params[key] for key in keys]

    configs = []

    if sweep_type == 'linear':
        setting_iter = zip(*values)
    elif sweep_type == 'grid':
        setting_iter = product(*values)
    
    for vals in setting_iter:
        conf = deepcopy(base_config)
        for k, v in zip(keys, vals):
            conf = deep_set(conf, k, v)
        configs.append(conf)
    
    return configs
