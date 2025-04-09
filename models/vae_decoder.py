# %%writefile /content/Product-design-GenAi-XAI/models/vae_decoder.py
import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_res=256):
        super().__init__()

        # Match with encoder's final output: 512 * 4 * 4
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        layers = [
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 4 → 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64 → 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        ]

        if out_res == 256:
            layers += [
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 → 256
            ]
        elif out_res == 224:
            layers += [
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 → 256
                nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
            ]

        layers += [
            nn.Tanh()  # Use Tanh because inputs are normalized to [-1, 1]
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.decoder(x)
