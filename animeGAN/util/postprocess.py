from pathlib import Path
import cv2
import torch
from torchvision import transforms
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def denorm(img_batch: torch.Tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    invert_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    imgs = invert_normalize(img_batch) * 255
    return imgs

def tensor2numpy(img_batch):
    return img_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

def RGB2BGR(img_batch):
    # inplace
    for i in range(len(img_batch)):
        img_batch[i] = cv2.cvtColor(img_batch[i], cv2.COLOR_RGB2BGR)
    return img_batch

def add_left_caption(img, caption_prefix, caption):
    H, W, C = img.shape
    padding = 100
    img = cv2.copyMakeBorder(img, 0, 0, padding, 0, cv2.BORDER_CONSTANT, value=WHITE)
    if caption_prefix:
        img = cv2.putText(img, str(caption_prefix), org=(20, H//2 - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                          fontScale=0.5, color=BLACK, thickness=1)
    if caption:
        img = cv2.putText(img, str(caption), org=(20, H // 2 + 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                          fontScale=0.5, color=BLACK, thickness=1)
    return img

class ImageGrid:
    def __init__(self, output_fpath: Path):
        self.image_rows = []
        self.output_fpath = output_fpath

    def add_row(self, img_batch: torch.Tensor, caption_prefix=None, caption=None):
        img_row = np.concatenate(RGB2BGR(tensor2numpy(denorm(img_batch))), axis=1).round().astype(int)
        img_row = add_left_caption(img_row, caption_prefix, caption)
        self.image_rows.append(img_row)

    def write(self):
        output_img = np.concatenate(self.image_rows)
        cv2.imwrite(str(self.output_fpath), output_img)
