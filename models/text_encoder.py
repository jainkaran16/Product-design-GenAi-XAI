import sys
sys.path.append("/content/Product-design-GenAi-XAI")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import CLIPTokenizer, CLIPTextModel
from models.vae_encoder import VAEEncoder
from utils.dataset_loader import ImageCaptionDataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CaptionProjector
class CaptionProjector(nn.Module):
    def __init__(self, latent_dim, finetune_clip=False):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.projector = nn.Linear(512, latent_dim)

        if not finetune_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, captions):
        tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
        clip_output = self.clip_model(**tokens)
        text_emb = clip_output.last_hidden_state[:, 0, :]
        return self.projector(text_emb)

# 2. Latent Fusion Module
class LatentFusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, image_latent, text_latent):
        return self.fusion(torch.cat((image_latent, text_latent), dim=1))

# 3. Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# 4. KL Divergence Loss
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

# 5. Training Function (✨ wrap everything here)
def train_text_encoder():
    print("🔥 Starting text encoder training...")

    latent_dim = 128

    # Load pretrained VAE encoder
    encoder = VAEEncoder(latent_dim).to(device)
    encoder.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/best_vae_encoder.pth"))
    encoder.eval()

    # Init models
    caption_projector = CaptionProjector(latent_dim, finetune_clip=True).to(device)
    latent_fusion = LatentFusion(latent_dim).to(device)

    # Optimizer & Scheduler
    params = list(filter(lambda p: p.requires_grad, caption_projector.parameters())) + list(latent_fusion.parameters())
    optimizer = torch.optim.Adam(params, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    loss_fn = nn.MSELoss()
    epochs = 10

    # Dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    full_dataset = ImageCaptionDataset(
        root_dir="/content/data/Furniture Dataset",
        transform=transform,
        use_caption=True
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training Loop
    best_loss = float('inf')

    for epoch in range(epochs):
        caption_projector.train()
        latent_fusion.train()
        total_loss = 0.0

        for images, captions in train_loader:
            images = images.to(device)
            captions = list(captions)

            with torch.no_grad():
                mu, logvar, _ = encoder(images)
                image_latents = reparameterize(mu, logvar)

            text_latents = caption_projector(captions)
            fused_latents = latent_fusion(image_latents, text_latents)

            recon_loss = loss_fn(fused_latents, image_latents)
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + 0.001 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        caption_projector.eval()
        latent_fusion.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = list(captions)

                mu, logvar, _ = encoder(images)
                image_latents = reparameterize(mu, logvar)

                text_latents = caption_projector(captions)
                fused_latents = latent_fusion(image_latents, text_latents)

                recon_loss = loss_fn(fused_latents, image_latents)
                kl_loss = kl_divergence(mu, logvar)
                val_loss += (recon_loss + 0.001 * kl_loss).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(caption_projector.state_dict(), "best_caption_projector.pth")
            torch.save(latent_fusion.state_dict(), "best_latent_fusion.pth")
            print(f"✅ Best models saved at epoch {epoch+1} with val_loss {avg_val_loss:.4f}")

    print("🎯 Text encoder training complete.")

# 6. Entry Point 🔥
if __name__ == "__main__":
    train_text_encoder()
