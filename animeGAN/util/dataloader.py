import logging
from pathlib import Path
import math
import copy
import torch
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

"""
Currently all training and style images are 256x256x3
TODO:
1. Add resize to convert real and anime to 256x256
"""

class AnimeGanDataset(Dataset):
    def __init__(self, image_folder, anime_folder, required_num_images=None, transform=None):
        self.image_folder = Path(image_folder)
        self.anime_folder = anime_folder
        self.transform = transform
        self.img_fpaths = [f for f in self.image_folder.iterdir()]
        self.num_images = len(self.img_fpaths)
        self.generate_smoothed_grayscale = False

        # create kernels for smoothing transform
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gauss = cv2.getGaussianKernel(kernel_size, 0)
        gauss = gauss * gauss.transpose(1, 0)
        self.smooth_params = dict(kernel_size=kernel_size, kernel=kernel, gaussian_kernel=gauss)

        # if required_num_images is specified, need to cycle through images for required_num_images by oversampling
        if required_num_images is not None:
            assert required_num_images >= self.num_images
            extra_cycles = int(math.ceil(required_num_images / self.num_images)) - 1
            img_fpaths = copy.copy(self.img_fpaths)
            for i in range(extra_cycles):
                fpaths = copy.copy(self.img_fpaths)
                np.random.shuffle(fpaths)
                img_fpaths += fpaths

            self.img_fpaths = img_fpaths[:required_num_images]

    def __len__(self):
        return len(self.img_fpaths)

    def load_image_as_rgb(self, fpath: str) -> Image:
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)  # to PIL Image
        return img

    def load_image_as_grayscale(self, fpath: str) -> Image:
        img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)  # 1 channel
        img_gray = np.asarray([img_gray, img_gray, img_gray])  # turn to 3 channel
        img_gray = np.transpose(img_gray, (1, 2, 0))  # transpose so its cv2 H x W x C
        img = Image.fromarray(img_gray)
        return img

    def _load_image_as_smoothed_gray(self, fpath: str) -> Image:
        img_bgr = cv2.imread(fpath)
        img_gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        kernel_size = self.smooth_params['kernel_size']
        kernel = self.smooth_params['kernel']
        gaussian_kernel = self.smooth_params['gaussian_kernel']

        img_bgr_pad = np.pad(img_bgr, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        edges = cv2.Canny(img_gray, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(img_bgr)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(img_bgr_pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gaussian_kernel))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(img_bgr_pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gaussian_kernel))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(img_bgr_pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gaussian_kernel))

        img = Image.fromarray(gauss_img)
        return img

    def load_image_as_smoothed_gray(self, fpath):
        if self.generate_smoothed_grayscale:  # set to False to not use because this is slow
            return self._load_image_as_smoothed_gray(fpath)
        else:
            # Load the preprocessed smooth image instead
            fpath = Path(fpath)
            fpath = fpath.parent.parent / 'smooth' / fpath.name  # load smooth version
            fpath = str(fpath)
            return self.load_image_as_grayscale(fpath)

    def load_image(self, fpath: Path):
        fpath = str(fpath)
        img = self.load_image_as_rgb(fpath)
        if self.anime_folder:
            # return list of 3 PIL images: original, grayscale and smoothed
            img_gray = self.load_image_as_grayscale(fpath)
            img_smooth_gray = self.load_image_as_smoothed_gray(fpath)

            return [img, img_gray, img_smooth_gray]
        else:
            return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fpath = self.img_fpaths[idx]
        image = self.load_image(fpath)

        # TODO: paper's code normalizes to [-1, 1] pytorch toTensor() normlizes to [0, 1]. Might cause results to be different
        if self.transform:
            if self.anime_folder:
                # image is actually 3 sets of images
                image = torch.stack([self.transform(img) for img in image])  # 3 x C x H x W
            else:
                image = self.transform(image)  # C x H x W
        return image


def get_dataloader(path: str, anime_folder: bool, batch_size: int, input_size: int, required_num_images=None, shuffle=True) -> DataLoader:

    # Transform train data with augmentation
    transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # pytorch vgg mean
        ]
    )

    # Get dataset
    dataset = AnimeGanDataset(path, anime_folder=anime_folder, required_num_images=required_num_images, transform=transform)

    # Train loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    log.info(f'Created dataloader from {path}')

    return dataloader


