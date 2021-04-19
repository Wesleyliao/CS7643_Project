import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, out_channel, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.convs(input)
        return output
