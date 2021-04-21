import torch
import torch.nn as nn

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
huber_loss = nn.SmoothL1Loss()

def content_loss(vgg, generated_images, real_images):
    gen_features = vgg(generated_images)
    real_features = vgg(real_images)
    return l1_loss(real_features, gen_features)

def style_loss(vgg, generated_images, anime_gray_images):
    def gram_matrix(features):
        N, C, H, W = features.shape
        F = features.view(N, C, H * W)  # batch x channels x features
        gram = torch.bmm(F, F.transpose(1, 2))
        # normalize
        gram /= H * W * C
        return gram

    gen_features = vgg(generated_images)
    anime_gray_features = vgg(anime_gray_images)
    gen_gram = gram_matrix(gen_features)
    anime_gray_gram = gram_matrix(anime_gray_features)
    return l1_loss(gen_gram, anime_gray_gram)


def color_loss(generated_images, real_images):
    def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
        # RGB image to YUV
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        out = torch.stack([y, u, v], dim=1)
        return out

    gen = rgb_to_yuv(generated_images)
    real = rgb_to_yuv(real_images)

    loss = l1_loss(gen[:, 0, :, :], real[:, 0, :, :]) + \
           huber_loss(gen[:, 1, :, :], real[:, 1, :, :]) + \
           huber_loss(gen[:, 2, :, :], real[:, 2, :, :])
    return loss

def total_variation_loss(img, device):
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]

    loss = l2_loss(dh, torch.zeros(dh.size()).to(device)) + \
           l2_loss(dw, torch.zeros(dw.size()).to(device))
    return loss
