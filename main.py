import click
import yaml

# Read config
CONFIG_PATH = './config/test.yml'
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)

def train():
    pass

def test():
    pass

@click.command()
@click.option(
    '--train',
    default=False,
    is_flag=True,
    help='Train model',
)
@click.option('--test', default=False, is_flag=True, help='Run test')
def main(train, test):
    """Train and test GAN model."""

    print(f'Config: \n{CONFIG}')
    
    if train:
        print('train...')
    if test:
        print('test...')