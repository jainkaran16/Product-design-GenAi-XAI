import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 32x32
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 64x64
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # 256x256
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 16, 16)
        return self.decoder(x)
