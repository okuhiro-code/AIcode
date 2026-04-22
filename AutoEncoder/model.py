import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, 5, 2, 2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2,
                               padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 64, 5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(64, 32, 5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 16, 5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(16, 1, 5, stride=2,
                               padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out
