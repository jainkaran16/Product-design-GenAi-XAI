import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels=4, cond_dim=512, base_channels=64):
        super().__init__()

        # Embed timestep and condition (both scalar to channel map)
        self.time_embed = nn.Linear(1, base_channels)
        self.cond_embed = nn.Linear(cond_dim, base_channels)

        # Downsampling encoder blocks
        self.enc1 = DoubleConv(in_channels + base_channels, base_channels)  # latent + t + cond
        self.enc2 = DoubleConv(base_channels, base_channels * 2)

        # Bottleneck
        self.middle = DoubleConv(base_channels * 2, base_channels * 4)

        # Decoder
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final projection
        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, cond):
        """
        Args:
            x: Noisy latent tensor (B, C, H, W)
            t: Time steps (B, 1) for timestep embedding
            cond: Condition (caption embedding, B, cond_dim)
        
        Returns:
            Predicted noise (B, C, H, W)
        """

        # Embed timestep and condition (projected to base_channels)
        t_embed = self.time_embed(t.unsqueeze(-1))  # [B, base_channels]
        c_embed = self.cond_embed(cond)             # [B, base_channels]

        # Expand and concatenate along channel axis
        t_channel = t_embed.unsqueeze(-1).unsqueeze(-1).expand_as(x[:, :1])  # [B, base_channels, H, W]
        c_channel = c_embed.unsqueeze(-1).unsqueeze(-1).expand_as(x[:, :1])  # [B, base_channels, H, W]
        
        # Concatenate input with condition and timestep embeddings
        x = torch.cat([x, t_channel, c_channel], dim=1)

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)

        # Bottleneck
        x3 = self.middle(x2)

        # Decoder with skip connections
        x = self.dec2(x3 + x2)
        x = self.dec1(x + x1)

        # Final output (predict noise or denoised latent)
        return self.final(x)
