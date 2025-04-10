import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder
from models.text_encoder import CaptionProjector
from models.unet import UNetModel
from diffusion.scheduler import LinearNoiseScheduler
from utils.dataset_loader import ImageCaptionDataset

# Configs
latent_dim = 512
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 30

# Load Models
vae_encoder = VAEEncoder().to(device)
vae_decoder = VAEDecoder().to(device)
caption_encoder = CaptionProjector(latent_dim=latent_dim).to(device)
unet = UNetModel(in_channels=4, cond_dim=latent_dim).to(device)
scheduler = LinearNoiseScheduler()

# Dataset
dataset = ImageCaptionDataset("data/train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = optim.Adam(unet.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Training loop
vae_encoder.eval()
caption_encoder.eval()
for epoch in range(epochs):
    for images, captions in dataloader:
        images = images.to(device)
        noise = torch.randn_like(images)
        latents = vae_encoder(images)
        timesteps = torch.randint(0, scheduler.timesteps, (images.size(0),), device=device).float() / scheduler.timesteps
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        cond = caption_encoder(captions)
        pred = unet(noisy_latents, timesteps, cond)
        loss = loss_fn(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

