import sys
sys.path.append("/content/Product-design-GenAi-XAI")
print("hi")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder
from models.text_encoder import CaptionProjector, LatentFusion  # Import both CaptionProjector and LatentFusion
from models.unet import UNetModel
from diffusion.scheduler import LinearNoiseScheduler
from utils.dataset_loader import ImageCaptionDataset
import os

# ==== Configs ====
latent_dim = 128
batch_size = 16  # Reduce if CUDA OOM
epochs = 20
lr = 1e-4
save_dir = "/content/Product-design-GenAi-XAI/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Check if the directory was created successfully
if os.path.exists(save_dir):
    print(f"Directory '{save_dir}' is ready.")
else:
    print(f"Failed to create directory '{save_dir}'.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Transforms ====
image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1]
])

# ==== Dataset ====
dataset = ImageCaptionDataset(root_dir="/content/data/Furniture Dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== Models ====
vae_encoder = VAEEncoder(latent_dim=latent_dim).to(device)
vae_decoder = VAEDecoder(latent_dim=latent_dim).to(device)

# Load Caption Projector and LatentFusion from .pth files
caption_encoder = CaptionProjector(latent_dim=latent_dim).to(device)
caption_encoder.load_state_dict(torch.load('/content/Product-design-GenAi-XAI/best_caption_projector.pth'))

latent_fusion = LatentFusion(latent_dim=latent_dim).to(device)  # Assuming LatentFusion is a class you have
latent_fusion.load_state_dict(torch.load('/content/drive/MyDrive/genai_checkpoints/best_latent_fusion.pth'))

unet = UNetModel(in_channels=4, cond_dim=latent_dim).to(device)
scheduler = LinearNoiseScheduler()

# Freeze VAE + Caption Encoder + Latent Fusion
vae_encoder.eval()
caption_encoder.eval()
latent_fusion.eval()

for p in vae_encoder.parameters():
    p.requires_grad = False
for p in caption_encoder.parameters():
    p.requires_grad = False
for p in latent_fusion.parameters():
    p.requires_grad = False

# ==== Optimizer and Loss ====
optimizer = optim.Adam(unet.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ==== Training Loop ====
print(">> Starting LDM Training...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        noise = torch.randn_like(images)

        with torch.no_grad():
            latents = vae_encoder(images)  # (B, 4, 32, 32) assuming latent space

        timesteps = torch.randint(0, scheduler.timesteps, (images.size(0),), device=device).long()
        timesteps_norm = timesteps.float() / scheduler.timesteps

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            cond = caption_encoder(captions)  # (B, latent_dim)

        # Use LatentFusion for combining text and image latents
        fused_latents = latent_fusion(latents, cond)

        pred_noise = unet(noisy_latents, timesteps_norm.unsqueeze(1), fused_latents)

        loss = loss_fn(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # Save after each epoch
    torch.save(unet.state_dict(), os.path.join(save_dir, f"unet_epoch{epoch+1}.pth"))

# Save Final
torch.save(unet.state_dict(), os.path.join(save_dir, "unet_final.pth"))
print(">> Training Finished. Final model saved!")
