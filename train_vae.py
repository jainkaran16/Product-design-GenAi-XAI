import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.dataset_loader import ImageCaptionDataset
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder

# ====================
# CONFIG
# ====================
LATENT_DIM = 256
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/content/Product-design-GenAi-XAI/data/Furniture Dataset"
CHECKPOINT_DIR = "/content/Product-design-GenAi-XAI/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ====================
# REPARAMETERIZATION
# ====================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ====================
# LOSS FUNCTIONS
# ====================
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_div  # scale KL term

# ====================
# LOAD DATA
# ====================
dataset = ImageCaptionDataset(DATASET_PATH, image_size=IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# INIT MODELS + OPTIMIZER
# ====================
encoder = VAEEncoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# ====================
# TRAINING LOOP
# ====================
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()

    for i, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)

        mu, logvar = encoder(images)
        z = reparameterize(mu, logvar)
        recon_images = decoder(z)

        loss = vae_loss_function(recon_images, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}] Loss: {loss.item():.4f}")
            save_image(recon_images[:4], f"{CHECKPOINT_DIR}/recon_epoch{epoch+1}_batch{i}.png")

    # Save model checkpoints after every epoch
    torch.save(encoder.state_dict(), f"{CHECKPOINT_DIR}/vae_encoder_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/vae_decoder_epoch{epoch+1}.pth")
