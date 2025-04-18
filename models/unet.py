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

        self.base_channels = base_channels

        # Embeddings
        self.time_embed = nn.Linear(1, base_channels)
        self.cond_embed = nn.Linear(cond_dim, base_channels)

        # Encoder
        self.enc1 = DoubleConv(in_channels + 2 * base_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)

        # Bottleneck
        self.middle = DoubleConv(base_channels * 2, base_channels * 4)

        # Skip connection projection (to match channels before addition)
        self.x2_proj = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)
        self.x1_proj = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)

        # Decoder
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final output - Changed to output 128 channels as 2D (B, 128)
        self.final = nn.Conv2d(base_channels, 128, kernel_size=1)  # Still keeping this, output will be (B, 128)

    def forward(self, x, t, cond):
        """
        Args:
            x: Noisy latent tensor (B, C, H, W)
            t: Timestep (B, 1)
            cond: Conditioning (B, cond_dim)
        Returns:
            Predicted noise (B, 128)
        """
        B, _, H, W = x.shape

        # Embeddings to channels
        t_embed = self.time_embed(t).view(B, -1, 1, 1).expand(B, self.base_channels, H, W)
        c_embed = self.cond_embed(cond).view(B, -1, 1, 1).expand(B, self.base_channels, H, W)

        # Ensure x is properly shaped (match the number of input channels to the model)
        x = x.expand(B, 4, H, W)

        # Concatenate x with t and cond embeddings
        x = torch.cat([x, t_embed, c_embed], dim=1)  # [B, 4 + 2*base_channels, H, W]

        # Encoder
        x1 = self.enc1(x)  # [B, base_channels, H, W]
        x2 = self.enc2(x1)  # [B, base_channels*2, H, W]

        # Bottleneck
        x3 = self.middle(x2)  # [B, base_channels*4, H, W]

        # Decoder with skip connections (project + add)
        x = self.dec2(x3 + self.x2_proj(x2))  # [B, base_channels*2, H, W]
        x = self.dec1(x + self.x1_proj(x1))   # [B, base_channels, H, W]

        # Output - Predicted noise, updated to output shape (B, 128)
        # Here, we take the average of the spatial dimensions to reduce it to 2D
        x = self.final(x)  # [B, 128, H, W]

        # Now reduce the spatial dimensions (H, W) to a single vector per image
        x = x.view(B, 128, -1)  # Flatten H and W into a single dimension (B, 128, H*W)
        x = torch.mean(x, dim=2)  # Mean across the spatial dimensions (H*W) to get (B, 128)

        return x  # Now the output is [B, 128]
