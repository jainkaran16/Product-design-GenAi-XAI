import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64x64
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 32x32
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# 16x16
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)
