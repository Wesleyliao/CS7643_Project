import torch
import torch.nn as nn


class Conv2D(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=3, padding=1, bias=False, padding_mode='zeros',
                 spectral_norm=True):
        pad_layer = {
            "zeros": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if (kernel_size - stride) % 2 == 0:
            pad_top = padding
            pad_bottom = padding
            pad_left = padding
            pad_right = padding
        else:
            pad_top = padding
            pad_bottom = kernel_size - stride - pad_top
            pad_left = padding
            pad_right = kernel_size - stride - pad_left

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        layers = [pad_layer[padding_mode](padding)]
        if spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=bias)))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=bias))

        super().__init__(*layers)


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, num_discriminator_layers=3, spectral_norm=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_discriminator_layers = num_discriminator_layers
        self.spectral_norm = spectral_norm

        # block A
        layers = [
            Conv2D(in_channel, 32, 3, 1, 1, bias=False, spectral_norm=spectral_norm),
            nn.LeakyReLU(0.2, True),
        ]

        # Iterative block B
        in_ch = 32
        out_ch = in_ch
        for i in range(num_discriminator_layers):
            layers += [
                Conv2D(in_ch, out_ch * 2, 3, 2, 1, bias=False, spectral_norm=spectral_norm),
                nn.LeakyReLU(0.2, True),

                Conv2D(out_ch * 2, out_ch * 4, 3, 1, 1, bias=False, spectral_norm=spectral_norm),
                nn.InstanceNorm2d(out_ch * 4),
                nn.LeakyReLU(0.2, True),
            ]
            in_ch = out_ch * 4
            out_ch *= 2

        # Block C
        layers += [
            Conv2D(in_ch, out_ch * 2, 3, 1, 1, bias=False, spectral_norm=spectral_norm),
            nn.InstanceNorm2d(out_ch * 2),
            nn.LeakyReLU(0.2, True),

            Conv2D(out_ch * 2, out_channel, 3, 1, 1, bias=False, spectral_norm=spectral_norm),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return output
