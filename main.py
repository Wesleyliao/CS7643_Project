import logging
import sys

import click
import torch
import yaml

from util.dataloader import get_dataloader

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s][%(module)s:%(funcName)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    level=logging.INFO,
)


# Open config as global variable
CONFIG_PATH = './config/default.yml'
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)


def train():
    pass


def test():
    pass


@click.command()
@click.option('--train', default=False, is_flag=True, help='Train model')
@click.option('--test', default=False, is_flag=True, help='Run test')
def main(train, test):
    """Train and test GAN model."""

    log.info("Starting run...")

    # print(f':::Running with config::: \n{yaml.dump(CONFIG, default_flow_style=False)}\n')

    # Check for GPU
    if torch.cuda.is_available():
        log.info(f'CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        log.info('CUDA not available.')

    # Get dataloaders
    anime_loader = get_dataloader(CONFIG['danbooru_path'], CONFIG['batch_size'])
    ffhq_loader = get_dataloader(CONFIG['ffhq_path'], CONFIG['batch_size'])

    # Test dataloader
    data, label = next(iter(anime_loader))
    log.info(f'Anime shapes {data.size()}, {label.size()}')

    data, label = next(iter(ffhq_loader))
    log.info(f'FFHQ shapes {data.size()}, {label.size()}')

    if train:
        print('train...')
    if test:
        print('test...')


if __name__ == '__main__':
    main()
