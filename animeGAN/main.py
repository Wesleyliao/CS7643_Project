import logging
import sys
import tqdm
import click
import torch
import yaml
import torch.nn as nn

from animeGAN.util.dataloader import get_dataloader
from animeGAN.util.loss import content_loss, style_loss, color_loss, total_variation_loss
from animeGAN.models.vgg import VGG19
from animeGAN.models.generator import Generator
from animeGAN.models.discriminator import Discriminator

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s][%(module)s:%(funcName)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    level=logging.INFO,
)


# Open config as global variable
CONFIG_PATH = './animeGAN/config/default.yml'
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)


def train(real_img_loader, anime_img_loader):
    # extract training params from yaml config
    epochs = CONFIG['epochs']
    init_epoch = CONFIG['init_epoch']
    batch_size = CONFIG['batch_size']
    # learning rates
    lr_g_init = CONFIG['lr_init']
    lr_g = CONFIG['lr_generator']
    lr_d = CONFIG['lr_discriminator']
    # weights
    g_real_weight = CONFIG['generator_is_anime_weight']
    g_content_weight = CONFIG['generator_content_weight']
    g_style_weight = CONFIG['generator_style_weight']
    g_color_weight = CONFIG['generator_color_weight']
    g_tv_weight = CONFIG['generator_tv_weight']
    d_real_weight = CONFIG['discriminator_is_anime_weight']
    d_fake_weight = CONFIG['discriminator_is_not_anime_weight']
    d_gray_weight = CONFIG['discriminator_gray_weight']
    d_edge_weight = CONFIG['discriminator_edge_weight']
    # vgg
    vgg_pretrain_weights = CONFIG['vgg19_pretrained_weights']
    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterions
    BCE_loss = nn.BCELoss()


    # set params
    start_epoch = 0
    generator = Generator()
    discriminator = Discriminator()
    vgg = VGG19(init_weights=vgg_pretrain_weights, feature_mode=True)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g_init, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # move models to gpu
    generator.to(device)
    discriminator.to(device)
    vgg.to(device)
    # set model modes
    generator.train()
    discriminator.train()
    vgg.eval()

    # training progress trackers
    init_training_mean_loss = []
    gan_training_mean_loss = []

    for epoch in range(start_epoch, epochs):
        for real_batch, anime_imgs_batch in zip(real_img_loader, anime_img_loader):
            # extract the 3 diff anime images from anime batch
            anime_batch, anime_gray_batch, anime_smooth_gray_batch = anime_imgs_batch.chunk(3, dim=1)
            # squeeze to remove the dim -> should now be B x C x H x W
            anime_batch = anime_batch.squeeze()
            anime_gray_batch = anime_gray_batch.squeeze()
            anime_smooth_gray_batch = anime_smooth_gray_batch.squeeze()
            # move train data to gpu
            real_batch = real_batch.to(device)
            anime_batch = anime_batch.to(device)
            anime_gray_batch = anime_gray_batch.to(device)
            anime_smooth_gray_batch = anime_smooth_gray_batch.to(device)
            # real if image is anime else fake (i.e. real images are fake)
            real = torch.ones(real_batch.size()).to(device)
            fake = torch.zeros(real_batch.size()).to(device)

            if epoch < init_epoch:
                # pre train generator using just content loss
                optimizer_g.zero_grad()

                generated_images = generator(real_batch)
                reconstruction_loss = content_loss(vgg, generated_images, real_batch)

                reconstruction_loss.backward()
                optimizer_g.step()
            else:
                # animeGAN training
                if epoch == init_epoch:
                    optimizer_g.param_groups[0]['lr'] = lr_g

                generated_images = generator(real_batch)

                """
                animeGAN paper -> alternatives discriminator
                GAN paper -> run discriminator for k batches before generating once
                """
                # train discriminator
                optimizer_d.zero_grad()

                d_generated_images = discriminator(generated_images)
                d_anime_images = discriminator(anime_batch)
                d_anime_gray_images = discriminator(anime_gray_batch)
                d_anime_smooth_gray_images = discriminator(anime_smooth_gray_batch)
                discriminator_loss = BCE_loss(d_anime_images, real) * d_real_weight + \
                                     BCE_loss(d_generated_images, fake) * d_fake_weight + \
                                     BCE_loss(d_anime_gray_images, fake) * d_gray_weight + \
                                     BCE_loss(d_anime_smooth_gray_images, fake) * d_edge_weight

                discriminator_loss.backward()
                optimizer_d.step()

                # train generator
                optimizer_g.zero_grad()

                d_generated_images = discriminator(generated_images)
                generator_loss = BCE_loss(d_generated_images, real) * g_real_weight + \
                                 content_loss(vgg, generated_images, real_batch) * g_content_weight + \
                                 style_loss(vgg, generated_images, anime_gray_batch) * g_style_weight + \
                                 color_loss(generated_images, real_batch) * g_color_weight + \
                                 total_variation_loss(generated_images, device) * g_tv_weight
                generator_loss.backward()
                optimizer_g.step()


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
    real_img_loader = get_dataloader(CONFIG['real_path'], CONFIG['batch_size'], CONFIG['input_size'])
    anime_img_loader = get_dataloader(CONFIG['anime_style_path'], CONFIG['batch_size'], CONFIG['input_size'], required_num_images=len(real_img_loader.dataset))

    # Test dataloader
    data = next(iter(real_img_loader))
    log.info(f'Real images shapes {data.size()}. Number of images {len(real_img_loader.dataset)}')

    data = next(iter(anime_img_loader))
    log.info(f'Anime images shapes {data.size()}. Number of images {len(anime_img_loader.dataset)}')

    if train:
        print('train...')
    if test:
        print('test...')


if __name__ == '__main__':
    main()
