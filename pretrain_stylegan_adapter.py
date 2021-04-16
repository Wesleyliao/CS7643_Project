import logging
import sys

import torch
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


def train(epoch, data_loader, model, optimizer):

    losses = AverageMeter()

    for idx, (data, _) in enumerate(data_loader):

        if torch.cuda.is_available():
            data = data.cuda()

        optimizer.zero_grad()

        # Forward pass
        anime_img, face_features, anime_features = model(data)

        criterion = torch.nn.MSELoss()
        loss = criterion(face_features, anime_features)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record loss
        losses.update(loss, anime_img.shape[0])

    log.info(f'Epoch {epoch} loss {losses.avg}')


def main():

    # Get portrait dataloader
    ffhq_loader = get_dataloader(CONFIG['ffhq_path'], CONFIG['batch_size'])

    # Get pretrained StyleGAN2 generator
    stylegan_model = get_pretrained_stylegan(CONFIG['pretrain_stylgan_path'])
    print(stylegan_model(torch.randn(2, 512)).size())

    # Get face feature extractor
    resnet_model = get_pretrained_resnet(CONFIG['pretrain_resnet_path'])
    print(resnet_model(torch.randn(2, 3, 160, 160)).size())

    # Get main model
    model = Frankenstein(resnet_model, stylegan_model)

    # Move model to cuda if it is available
    if torch.cuda.is_available():
        model = model.cuda()

    # Log trainable layers
    for param in model.parameters():
        if param.requires_grad:
            log.info(f'Trainable layer: {param.size()} {param.device}')

    # Setup
    best_model = None
    float('inf')
    optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])

    # Train
    log.info('Starting training...')
    for i in range(10):

        train(i, ffhq_loader, model, optimizer)

    # Save model
    torch.save(best_model.state_dict(), './checkpoints/pretrained_adapter.pth')


if __name__ == '__main__':
    main()
