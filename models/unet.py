import torch
import torch.nn as nn

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
        self.enc1 = DoubleConv(in_channels + 1 + 1, base_channels)  # +1 timestep, +1 condition
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.middle = DoubleConv(base_channels * 2, base_channels * 4)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.final = nn.Conv2d(base_channels, in_channels, 1)
        self.time_embed = nn.Linear(1, 1)
        self.cond_embed = nn.Linear(cond_dim, 1)

    def forward(self, x, t, cond):
        # Embed timestep and cond into extra channels
        t_embed = self.time_embed(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        c_embed = self.cond_embed(cond).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, t_embed.expand_as(x[:, :1]), c_embed.expand_as(x[:, :1])], dim=1)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.middle(x2)
        x = self.dec2(x3 + x2)
        x = self.dec1(x + x1)
        return self.final(x)

