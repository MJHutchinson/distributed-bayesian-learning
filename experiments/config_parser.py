import os
import sys
import json

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
    names = []

    if sweep_type == 'linear':
        setting_iter = zip(*values)
    elif sweep_type == 'grid':
        setting_iter = product(*values)
    
    for vals in setting_iter:
        conf = deepcopy(base_config)
        name = ''
        for k, v in zip(keys, vals):
            conf = deep_set(conf, k, v)
            name = name + f'_{k.split(".")[-1]}_{v}'
        name = name[1:]
        conf['results_dir'] = os.path.join(conf['results_dir'], name)
        configs.append(conf)
        names.append(name)
    
    return configs, names

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise(RuntimeError("Wrong arguments, use python config_parser.py <path_to_config>"))
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        sweep_config = json.load(f)

    configs, names = parse_sweep_config(sweep_config)

    os.makedirs(
        os.path.expandvars(
                os.path.join(sweep_config['sweep_config']['output_dir'], 'configs')
            )
    )

    for config, name in zip(configs, names):
        with open(
            os.path.expandvars(
                os.path.join(sweep_config['sweep_config']['output_dir'], 'configs', name + '.json')),
                'w'
            ) as f:
            json.dump(config, f)
