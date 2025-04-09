# %%writefile /content/Product-design-GenAi-XAI/models/vae_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_res=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 4 → 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 8 → 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16 → 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32 → 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64 → 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128 → 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Output conv layer
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.final_activation = nn.Tanh()
        self.out_res = out_res

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.decoder(x)
        x = self.out_conv(x)

        if self.out_res == 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return self.final_activation(x)
