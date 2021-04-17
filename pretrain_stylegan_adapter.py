import logging
import os
import sys

import click
import torch
import torchvision
import yaml

from models.frankenstein import Frankenstein
from pretrain import resnet_face_model
from pretrain import stylegan2_model
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
CONFIG_PATH = './config/pretrain_stylegan_adapter.yml'
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_pretrained_stylegan(path):

    log.info(f'Loading pre-trained StyleGAN-2 model {path}')
    model = stylegan2_model.load(path)

    return model


def get_pretrained_resnet(path):

    log.info(f'Loading pre-trained Inception Resnet face recognition model {path}')
    model = resnet_face_model.InceptionResnetV1(
        pretrained='vggface2', checkpoint_path=path
    )

    return model


def train(epoch, data_loader, model, optimizer, scheduler):

    losses = AverageMeter()

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    for idx, (data, _) in enumerate(data_loader):

        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.cuda()

        # Forward pass
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            anime_img, face_features, anime_features = model(data)
            criterion = torch.nn.MSELoss()
            loss = criterion(face_features, anime_features)

        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        # Update learning rate scheduler
        scheduler.step()

        # Record loss
        losses.update(loss, anime_img.shape[0])

        log.info(f'E{epoch}, iter {idx}, loss {loss}')

        # Save every 200
        if idx % 200 == 0:
            torch.save(model.state_dict(), CONFIG['save_path'])

    log.info(f'--- Epoch {epoch} loss {losses.avg} ---')


@click.command()
@click.option('--train', default=False, is_flag=True, help='Train model')
@click.option('--test', default=False, is_flag=True, help='Run test')
def main(train, test):

    # Get portrait dataloader
    ffhq_loader = get_dataloader(
        CONFIG['ffhq_path'], CONFIG['batch_size'], image_size=160
    )

    # Get pretrained StyleGAN2 generator
    stylegan_model = get_pretrained_stylegan(CONFIG['pretrain_stylgan_path'])
    print(stylegan_model(torch.randn(2, 512)).size())

    # Get face feature extractor
    resnet_model = get_pretrained_resnet(CONFIG['pretrain_resnet_path'])
    print(resnet_model(torch.randn(2, 3, 160, 160)).size())

    # Get main model
    model = Frankenstein(resnet_model, stylegan_model)

    # Load existing weights if they exist
    if os.path.isfile(CONFIG['save_path']):

        log.info(f"Loading existing weights from {CONFIG['save_path']}")
        # print('stuff', model.generator.G_synthesis.conv_blocks[7].conv_block[0])
        model.load_state_dict(torch.load(CONFIG['save_path']), strict=False)

    if train:

        # Move model to cuda if it is available
        if torch.cuda.is_available():
            model = model.cuda()

        # Log trainable layers
        for param in model.parameters():
            if param.requires_grad:
                log.info(f'Trainable layer: {param.size()} {param.device}')

        # Setup
        optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=CONFIG['lr'],
            max_lr=CONFIG['lr'] * 5,
            cycle_momentum=False,
        )

        # Train
        log.info('Starting training...')
        for i in range(10):
            train(i, ffhq_loader, model, optimizer, scheduler)

    if test:

        # Get single batch of data
        data, label = next(iter(ffhq_loader))
        # data = torch.randn(20, 3, 160, 160)

        # Evaluate
        model.eval()
        with torch.no_grad():
            anime_img, face_features, anime_features = model(data)

        # anime_img = stylegan_model(torch.randn(20, 512))

        # Save images
        torchvision.utils.save_image(data, os.path.join(CONFIG['out_ffhq_img_path']))
        torchvision.utils.save_image(
            anime_img, os.path.join(CONFIG['out_anime_img_path'])
        )


if __name__ == '__main__':
    main()
