import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, out_res=256):
        super(VAEDecoder, self).__init__()
        self.out_res = out_res
        self.latent_dim = latent_dim

        # Fully connected layer to project latent vector to feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder layers with skip connections
        self.dec1 = self.upconv_block(512, 384)  # 4x4 -> 8x8
        self.dec2 = self.upconv_block(384, 384)  # 8x8 -> 16x16
        self.dec3 = self.upconv_block(384, 256)  # 16x16 -> 32x32
        self.dec4 = self.upconv_block(256, 128)  # 32x32 -> 64x64
        self.dec5 = self.upconv_block(128, 64)   # 64x64 -> 128x128
        self.dec6 = self.upconv_block(64, 3, final_layer=True)  # 128x128 -> 256x256

    def upconv_block(self, in_channels, out_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()  # Output range [-1, 1]
            )

    def forward(self, z, skip_connections=None):
    # Project latent vector to feature map
      x = self.fc(z)
      x = x.view(x.size(0), 512, 4, 4)

      # Safe skip connections handling
      if skip_connections is None:
          skip_connections = [None] * 5  # total 5 skip connections used

      x = self.dec1(x)
      if skip_connections[0] is not None:
          x = x + skip_connections[0]  # 8x8

      x = self.dec2(x)
      if skip_connections[1] is not None:
          x = x + skip_connections[1]  # 16x16

      x = self.dec3(x)
      if skip_connections[2] is not None:
          x = x + skip_connections[2]  # 32x32

      x = self.dec4(x)
      if skip_connections[3] is not None:
          x = x + skip_connections[3]  # 64x64

      x = self.dec5(x)
      if skip_connections[4] is not None:
          x = x + skip_connections[4]  # 128x128

      x = self.dec6(x)  # 256x256

      return x