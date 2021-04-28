import torch
import torch.nn as nn

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
l1_loss = nn.L1Loss().to(device)
l2_loss = nn.MSELoss().to(device)
huber_loss = nn.SmoothL1Loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device)

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

def total_variation_loss(img):
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]

    loss = l2_loss(dh, torch.zeros(dh.size()).to(device)) + \
           l2_loss(dw, torch.zeros(dw.size()).to(device))
    return loss

def discriminator_loss(d_anime_images, d_generated_images, d_anime_gray_images, d_anime_smooth_gray_images,
                       d_real_weight, d_fake_weight, d_gray_weight, d_edge_weight,
                       gan_loss_type):
    # real if image is anime else fake (i.e. real images are fake)
    real = torch.ones(d_generated_images.size()).to(device)
    fake = torch.zeros(d_generated_images.size()).to(device)

    if gan_loss_type == 'gan':
        loss = bce_loss(d_anime_images, real) * d_real_weight + \
               bce_loss(d_generated_images, fake) * d_fake_weight + \
               bce_loss(d_anime_gray_images, fake) * d_gray_weight + \
               bce_loss(d_anime_smooth_gray_images, fake) * d_edge_weight
    elif gan_loss_type == 'lsgan':
        loss = l2_loss(d_anime_images, real) * d_real_weight + \
               l2_loss(d_generated_images, fake) * d_fake_weight + \
               l2_loss(d_anime_gray_images, fake) * d_gray_weight + \
               l2_loss(d_anime_smooth_gray_images, fake) * d_edge_weight

    return loss

def generator_loss(d_generated_images, gan_loss_type):
    real = torch.ones(d_generated_images.size()).to(device)

    if gan_loss_type == 'gan':
        loss = bce_loss(d_generated_images, real)
    elif gan_loss_type == 'lsgan':
        loss = l2_loss(d_generated_images, real)

    return loss
