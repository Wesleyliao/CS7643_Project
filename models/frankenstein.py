from torch import nn


class Frankenstein(nn.Module):
    def __init__(self, feature_model, generator):
        super().__init__()

        self.feature_model = feature_model
        self.generator = generator

        self.fc1 = nn.Linear(512, 512, bias=False)
        self.fc2 = nn.Linear(512, 512, bias=False)

    def forward(self, x):

        # x is (N, 3, 160, 160)
        pass
