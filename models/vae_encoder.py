import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()# final output of conv layers must match original model
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)


    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# encoder = VAEEncoder(latent_dim=256).to(device)
# encoder.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/vae_encoder_epoch10.pth"))
# encoder.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with torch.no_grad():
#     img = torch.randn(1, 3, 256, 256).to(device)
#     mu, logvar = encoder(img)
#     print(mu.shape, logvar.shape)
