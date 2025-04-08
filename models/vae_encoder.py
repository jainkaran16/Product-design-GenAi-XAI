import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim  # 👈 Store latent_dim for later use
        self.flattened_dim = None
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

        self.flattened_dim = None  # we'll compute this dynamically
        self.fc_mu = None
        self.fc_logvar = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        
        if self.flattened_dim is None:
            self.flattened_dim = x.shape[1]
            self.fc_mu = nn.Linear(self.flattened_dim, self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(self.flattened_dim, self.latent_dim).to(x.device)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
