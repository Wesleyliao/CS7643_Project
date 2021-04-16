from torch import nn
from torch.nn import functional as F


class Frankenstein(nn.Module):
    def __init__(self, feature_model, generator):
        super().__init__()

        self.feature_model = feature_model
        self.generator = generator

        for params in self.feature_model.parameters():
            params.requires_grad = False

        for params in self.generator.parameters():
            params.requires_grad = False

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):

        x = F.interpolate(x, size=160)
        # x is (N, 3, 160, 160)

        face_features = self.feature_model(x)

        # face_features is (N, 512)

        x = F.relu(face_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # x is (N, 512)

        anime_img = self.generator(x)

        # x is (N, 3, 512, 512)

        x = F.interpolate(anime_img, size=160)

        # x is (N, 3, 160, 160)

        anime_features = self.feature_model(x)

        # anime_features is (N, 512)
        # print(f'outputting', anime_img.size(), face_features.size(), anime_features.size())
        return anime_img, face_features, anime_features
