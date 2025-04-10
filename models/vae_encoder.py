import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers with skip connections
        self.enc1 = self.conv_block(3, 64)    # 256x256 -> 128x128
        self.enc2 = self.conv_block(64, 128)  # 128x128 -> 64x64
        self.enc3 = self.conv_block(128, 256) # 64x64 -> 32x32
        self.enc4 = self.conv_block(256, 384) # 32x32 -> 16x16
        self.enc5 = self.conv_block(384, 384) # 16x16 -> 8x8
        self.enc6 = self.conv_block(384, 512) # 8x8 -> 4x4

        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Forward pass with skip connections
        x1 = self.enc1(x)  # 128x128
        x2 = self.enc2(x1) # 64x64
        x3 = self.enc3(x2) # 32x32
        x4 = self.enc4(x3) # 16x16
        x5 = self.enc5(x4) # 8x8
        x6 = self.enc6(x5) # 4x4

        # Flatten and compute mean and log variance
        x6_flat = x6.view(x6.size(0), -1)
        mu = self.fc_mu(x6_flat)
        logvar = self.fc_logvar(x6_flat)

        # Return skip connections along with mu and logvar
        return mu, logvar, [x5, x4, x3, x2, x1]
