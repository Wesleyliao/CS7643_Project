import collections
import logging
from pathlib import Path
import sys
import time
import tqdm
import click
import torch
import yaml
import torch.nn as nn
import numpy as np
import pickle
import cv2

from animeGAN.util.dataloader import get_dataloader
from animeGAN.util.loss import *
from animeGAN.models.vgg import VGG19
from animeGAN.models.generator import Generator
from animeGAN.models.discriminator import Discriminator
from animeGAN.util.postprocess import ImageGrid
from animeGAN.util.pbar import get_pbar

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s][%(module)s:%(funcName)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    level=logging.INFO,
)

def load_checkpoint(fpath, generator, discriminator, optimizer_g, optimizer_d):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model is on gpu
    checkpoint = torch.load(fpath, map_location=device)
    last_finished_epoch = checkpoint['epoch']
    generator.load_state_dict(checkpoint['generator'])
    if 'discriminator' in checkpoint and discriminator:
        discriminator.load_state_dict(checkpoint['discriminator'])
    # optimizer is always cpu
    # checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
    # if 'optimizer_g' in checkpoint and optimizer_g:
    #     optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    # if 'optimizer_d' in checkpoint and optimizer_d:
    #     optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    start_epoch = last_finished_epoch + 1
    return start_epoch

def load_train_hists(fpath):
    with open(fpath, 'rb') as f:
        hist = pickle.load(f)
        return hist

def train(real_img_loader, anime_img_loader, eval_img_loader):
    # extract training params from yaml config
    train_epochs = CONFIG['train_epochs']
    init_epoch = CONFIG['init_epoch']
    batch_size = CONFIG['batch_size']
    # learning rates
    lr_g_init = CONFIG['lr_generator_init']
    lr_g = CONFIG['lr_generator']
    lr_d = CONFIG['lr_discriminator']
    # weights
    adv_weight = CONFIG['adversarial_weight']
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
    # gan
    gan_loss_type = CONFIG['gan_loss_type']
    # discriminator
    spectral_norm = CONFIG['spectral_norm']
    num_discriminator_layers = CONFIG['num_discriminator_layers']
    disc_train_freq = CONFIG['discriminator_train_freq']
    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fpath
    output_dir = Path(CONFIG['output_path'])
    checkpoint_dir = Path(CONFIG['checkpoint_path'])
    model_name = CONFIG['model_name']
    load_checkpoint_fpath = Path(CONFIG['load_checkpoint_fpath']) if CONFIG['load_checkpoint_fpath'] else None
    load_history = CONFIG['load_history']

    # model export fpath
    pretrain_hist_fpath = output_dir / f'{model_name}_pretrain_hist.pkl'
    train_hist_fpath = output_dir / f'{model_name}_train_hist.pkl'
    checkpoint_fpath = checkpoint_dir / f'{model_name}.pt'
    reconstruction_checkpoint_fpath = checkpoint_dir / f'{model_name}_recon.pt'
    # model params
    generator = Generator()
    discriminator = Discriminator(num_discriminator_layers=num_discriminator_layers, spectral_norm=spectral_norm)
    vgg = VGG19(init_weights=vgg_pretrain_weights, feature_mode=True)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_g_init = torch.optim.Adam(generator.parameters(), lr=lr_g_init, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    if load_checkpoint_fpath and load_checkpoint_fpath.exists():
        start_epoch = load_checkpoint(load_checkpoint_fpath, generator, discriminator, optimizer_g, optimizer_d)
        log.info(f'Loaded from checkpoint {load_checkpoint_fpath}')
        log.info(f'Continuing with starting epoch: {start_epoch + 1}')
    else:
        start_epoch = 0
    # training progress trackers
    pretrain_hist = collections.defaultdict(list)
    train_hist = collections.defaultdict(list)
    pretrain_hist['gan_loss_type'] = gan_loss_type
    train_hist['gan_loss_type'] = gan_loss_type
    if load_history:
        if (output_dir / f'{load_checkpoint_fpath.stem}_pretrain_hist.pkl').exists():
            pretrain_hist = load_train_hists(output_dir / f'{load_checkpoint_fpath.stem}_pretrain_hist.pkl')
        train_hist = load_train_hists(output_dir / f'{load_checkpoint_fpath.stem}_train_hist.pkl')
        log.info(f'Loaded existing training history')

    # move models to gpu
    generator.to(device)
    discriminator.to(device)
    vgg.to(device)
    # set model modes
    vgg.eval()
    # get eval images and add to img grid
    eval_freq = CONFIG['eval_freq']
    eval_batch = next(iter(eval_img_loader))
    eval_batch = eval_batch.to(device)
    # output image grid
    #TODO: load existing image into image grid so we can keep adding to it
    img_grid = ImageGrid(output_dir / 'val' / f'{model_name}_val.png')
    img_grid.add_row(eval_batch, 'Orig')
    # epochs
    total_epochs = start_epoch + train_epochs
    # print model architecture
    # log.info(f'Generator Architecture:\n{generator}')
    # log.info(f'Discriminator Architecture:\n{discriminator}')
    iter_num = 0
    for epoch in get_pbar(np.arange(start_epoch, total_epochs), desc='Epoch Progress'):
        pretrain = epoch < init_epoch
        pretrain_epoch = epoch + 1
        train_epoch = epoch - init_epoch + 1

        #####################
        # Train
        #####################
        generator.train()
        discriminator.train()

        epoch_start_time = time.time()
        recon_loss, gen_loss, disc_loss, batch_time = [], [], [], []
        pbar = get_pbar(zip(real_img_loader, anime_img_loader), total=len(real_img_loader), desc=f'Batch Progress [Epoch {epoch+1} / {total_epochs}]')
        for real_batch, anime_imgs_batch in pbar:
            batch_start_time = time.time()

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

            if pretrain:
                # pre train generator using just content loss
                optimizer_g_init.zero_grad()

                generated_images = generator(real_batch)
                reconstruction_loss = content_loss(vgg, generated_images, real_batch)

                reconstruction_loss.backward()
                optimizer_g_init.step()

                recon_loss.append(reconstruction_loss.item())
            else:
                # train discriminator every N training epochs
                if iter_num % disc_train_freq == 0:
                    optimizer_d.zero_grad()

                    generated_images = generator(real_batch)

                    d_generated_images = discriminator(generated_images)
                    d_anime_images = discriminator(anime_batch)
                    d_anime_gray_images = discriminator(anime_gray_batch)
                    d_anime_smooth_gray_images = discriminator(anime_smooth_gray_batch)

                    d_loss = discriminator_loss(d_anime_images, d_generated_images, d_anime_gray_images,
                                                d_anime_smooth_gray_images, d_real_weight, d_fake_weight, d_gray_weight,
                                                d_edge_weight, gan_loss_type) * adv_weight

                    d_loss.backward()
                    optimizer_d.step()

                    disc_loss.append(d_loss.item())

                # train generator
                optimizer_g.zero_grad()

                generated_images = generator(real_batch)

                d_generated_images = discriminator(generated_images)

                g_loss = generator_loss(d_generated_images, gan_loss_type) * adv_weight + \
                         content_loss(vgg, generated_images, real_batch) * g_content_weight + \
                         style_loss(vgg, generated_images, anime_gray_batch) * g_style_weight + \
                         color_loss(generated_images, real_batch) * g_color_weight + \
                         total_variation_loss(generated_images) * g_tv_weight

                g_loss.backward()
                optimizer_g.step()

                gen_loss.append(g_loss.item())

            # save checkpoint after every 10 iterations
            if iter_num % 10 == 0:
                checkpoint = dict(epoch=epoch, generator=generator.state_dict(), discriminator=discriminator.state_dict(),
                                  optimizer_g=optimizer_g.state_dict(), optimizer_d=optimizer_d.state_dict(),
                                  config=CONFIG, iter_num=iter_num)
                torch.save(checkpoint, checkpoint_fpath)

            iter_num += 1
            batch_time.append(time.time() - batch_start_time)

        batch_time = np.mean(batch_time)
        train_time = time.time() - epoch_start_time
        if pretrain:
            recon_loss = np.mean(recon_loss)

            pretrain_hist['epoch'].append(pretrain_epoch)
            pretrain_hist['recon_loss'].append(recon_loss)
            pretrain_hist['epoch_time'].append(train_time)
            pretrain_hist['batch_time'].append(batch_time)
            print(f'[{pretrain_epoch}/{init_epoch} Pretrain]\t\tRecon loss {recon_loss:.02f}\t\t'
                  f'Epoch time {train_time:.02f}\t\tBatch time {batch_time:.02f}')

            # save pre train hist to file just once
            with open(pretrain_hist_fpath, 'wb') as f:
                pickle.dump(pretrain_hist, f)

            # save generator weights from reconstruction pretrain
            checkpoint = dict(epoch=epoch, generator=generator.state_dict(), optimizer_g=optimizer_g.state_dict(),
                              optimizer_d=optimizer_d.state_dict(), config=CONFIG)
            torch.save(checkpoint, reconstruction_checkpoint_fpath)
        else:
            disc_loss = np.mean(disc_loss)
            gen_loss = np.mean(gen_loss)

            train_hist['epoch'].append(train_epoch)
            train_hist['disc_loss'].append(disc_loss)
            train_hist['gen_loss'].append(gen_loss)
            train_hist['epoch_time'].append(train_time)
            train_hist['batch_time'].append(batch_time)
            print(f'[{train_epoch}/{total_epochs - init_epoch} Train]\t\tDisc loss {disc_loss:.02f}\t\t'
                  f'Gen loss {gen_loss:.02f}\t\tEpoch time {train_time:.02f}\t\tBatch time {batch_time:.02f}')

            # save train hist to file
            with open(train_hist_fpath, 'wb') as f:
                pickle.dump(train_hist, f)

            # save at the end of each epoch
            checkpoint = dict(epoch=epoch, generator=generator.state_dict(), discriminator=discriminator.state_dict(),
                              optimizer_g=optimizer_g.state_dict(), optimizer_d=optimizer_d.state_dict(), config=CONFIG,
                              iter_num=iter_num)
            torch.save(checkpoint, checkpoint_fpath)

        #####################
        # Eval
        #####################
        if epoch % eval_freq == 0 or epoch == total_epochs - 1 or epoch == init_epoch - 1:
            generator.eval()

            with torch.no_grad():
                generated_images = generator(eval_batch)
                # add to img grid
                if pretrain:
                    img_grid.add_row(generated_images, 'pretrain', pretrain_epoch)
                else:
                    img_grid.add_row(generated_images, 'train', train_epoch)

                # save images
                img_grid.write()


def test(eval_img_loader):
    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fpath
    output_dir = Path(CONFIG['output_path'])
    model_name = CONFIG['model_name']
    load_checkpoint_fpath = Path(CONFIG['load_checkpoint_fpath']) if CONFIG['load_checkpoint_fpath'] else None

    # model params
    generator = Generator()
    if load_checkpoint_fpath and load_checkpoint_fpath.exists():
        _ = load_checkpoint(load_checkpoint_fpath, generator, None, None, None)
        log.info(f'Loaded from checkpoint {load_checkpoint_fpath}')
    else:
        return

    # move models to gpu
    generator.to(device)
    eval_batch = next(iter(eval_img_loader))
    eval_batch = eval_batch.to(device)
    # output image grid
    img_grid = ImageGrid(output_dir / 'val' / f'{model_name}_val.png')
    img_grid.add_row(eval_batch, 'Orig')

    #####################
    # Eval
    #####################
    generator.eval()

    with torch.no_grad():
        generated_images = generator(eval_batch)
        # add to img grid
        img_grid.add_row(generated_images, 'eval', None)
        # save images
        img_grid.write()

def test2(eval_img_loader):
    from animeGAN.util.postprocess import RGB2BGR, tensor2numpy, denorm
    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fpath
    output_dir = Path(CONFIG['output_path'])
    load_checkpoint_fpath = Path(CONFIG['load_checkpoint_fpath']) if CONFIG['load_checkpoint_fpath'] else None

    # model params
    generator = Generator()
    if load_checkpoint_fpath and load_checkpoint_fpath.exists():
        _ = load_checkpoint(load_checkpoint_fpath, generator, None, None, None)
        log.info(f'Loaded from checkpoint {load_checkpoint_fpath}')
    else:
        return

    # move models to gpu
    generator.to(device)
    # output image grid
    export_dir = output_dir / 'val' / 'fid'

    #####################
    # Eval
    #####################
    generator.eval()

    with torch.no_grad():
        img_id = 0
        for eval_batch in get_pbar(eval_img_loader, total=len(eval_img_loader)):
            eval_batch = eval_batch.to(device)

            generated_images = generator(eval_batch)
            generated_images = RGB2BGR(tensor2numpy(denorm(generated_images)))
            for img in generated_images:
                p = export_dir / f'{img_id}.png'
                cv2.imwrite(str(p), img)

                img_id += 1

@click.command()
@click.option('--mode', type=click.Choice(['train', 'test', 'test2'], case_sensitive=False), default='train', help='Train or test mode')
@click.option('--debug-mode', is_flag=True, help='Specify if running in debug mode')
@click.option('--config-path', type=str, default='./animeGAN/config/train.yml', help='Path to config file')
def main_cli(mode, debug_mode, config_path):
    main(mode, debug_mode, config_path)


def main(mode, debug_mode, config_path):
    """Train and test GAN model."""

    log.info("Starting run...")

    # Open config as global variable
    with open(config_path, 'r') as stream:
        global CONFIG
        CONFIG = yaml.safe_load(stream)
        # override CONFIG with debug values for faster training and debugging
        if debug_mode:
            CONFIG['train_epochs'] = 2
            CONFIG['batch_size'] = 4
            CONFIG['init_epoch'] = 1
            CONFIG['eval_freq'] = 1
            required_num_images = 4
        else:
            required_num_images = None

    # Check for GPU
    if torch.cuda.is_available():
        log.info(f'CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        log.info('CUDA not available.')

    # Get dataloaders
    if mode.lower() == 'train':
        real_img_loader = get_dataloader(CONFIG['real_path'], False, CONFIG['batch_size'], CONFIG['input_size'], required_num_images=required_num_images)
        anime_img_loader = get_dataloader(CONFIG['anime_style_path'], True, CONFIG['batch_size'], CONFIG['input_size'],
                                          required_num_images=len(real_img_loader.dataset),
                                          generate_smoothed_grayscale=CONFIG['generate_smoothed_grayscale'])
        data = next(iter(real_img_loader))
        log.info(f'Real images shapes {data.size()}. Number of images {len(real_img_loader.dataset)}')

        data = next(iter(anime_img_loader))
        log.info(f'Anime images shapes {data.size()}. Number of images {len(anime_img_loader.dataset)}')

    eval_img_loader = get_dataloader(CONFIG['eval_path'], False, 8, CONFIG['input_size'], shuffle=False)

    # Test dataloader
    data = next(iter(eval_img_loader))
    log.info(f'Eval images shapes {data.size()}. Number of images {len(eval_img_loader.dataset)}')

    if mode.lower() == 'train':
        train(real_img_loader, anime_img_loader, eval_img_loader)
    if mode.lower() == 'test':
        test(eval_img_loader)
    if mode.lower() == 'test2':
        test2(eval_img_loader)


if __name__ == '__main__':
    main_cli()
