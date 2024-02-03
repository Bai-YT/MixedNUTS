import torch
import numpy as np
import random

import yaml
import os


def outer_prod(tensor1, tensor2):
    """ Outer product of two tensors of arbitrary shapes. """
    # Get the number of dimensions for each tensor
    dims1 = len(tensor1.shape)
    dims2 = len(tensor2.shape)

    # Construct the einsum string
    # For tensor1, we use 'abcd...' and for tensor2, we use 'efgh...'
    einsum_str = (
        ''.join(chr(i + 97) for i in range(dims1)) + ',' +
        ''.join(chr(i + 97 + dims1) for i in range(dims2)) + '->' +
        ''.join(chr(i + 97) for i in range(dims1 + dims2))
    )

    # Compute the outer product using einsum
    return torch.einsum(einsum_str, tensor1, tensor2)


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_yaml(filename):
    with open(filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def eval_configs(config):
    for key, val in config.items():
        if isinstance(val, str):
            if ("(" in val or "[" in val or val == 'None' or
                ("e-" in val and len(val) < 8)):
                try:
                    config[key] = eval(val)
                except:
                    print(f"Could not convert {val}.")
        elif isinstance(val, dict):
            eval_configs(val)
