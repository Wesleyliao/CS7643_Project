import logging
import sys

import torch
import yaml

from pretrain import stylegan2_model

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s][%(module)s:%(funcName)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    level=logging.INFO,
)


# Open config as global variable
CONFIG_PATH = './config/pretrain_stylegan_adapter.yml'
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)


def get_pickled_model(path):

    log.info(f'Loading existing model {path}')
    model = stylegan2_model.load(path)

    return model


def main():

    stylegan_model = get_pickled_model(CONFIG['pretrain_stylgan_path'])
    print(stylegan_model(torch.randn(1, 512)))


if __name__ == '__main__':
    main()
