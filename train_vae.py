import os
import torch
import clip
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import vgg16

from utils.dataset_loader import ImageCaptionDataset
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder

# ====================
# CONFIG
# ====================
LATENT_DIM = 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/content/data/Furniture Dataset"
CHECKPOINT_DIR = "/content/Product-design-GenAi-XAI/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ====================
# LOAD CLIP
# ====================
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ====================
# CUSTOM TRANSFORM
# ====================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ====================
# LOAD DATASET
# ====================
dataset = ImageCaptionDataset(DATASET_PATH, transform=transform, use_caption=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# MODELS
# ====================
encoder = VAEEncoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(latent_dim=LATENT_DIM).to(DEVICE)
caption_projector = nn.Linear(512, LATENT_DIM).to(DEVICE)

# ====================
# LATENT FUSION
# ====================
class LatentFusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z_img, z_txt):
        combined = torch.cat([z_img, z_txt], dim=1)
        return self.fusion(combined)

latent_fusion = LatentFusion(LATENT_DIM).to(DEVICE)

# ====================
# VGG FOR PERCEPTUAL LOSS
# ====================
vgg = vgg16(pretrained=True).features[:16].to(DEVICE).eval()
for param in vgg.parameters():
    param.requires_grad = False

# ====================
# LOSS
# ====================
def perceptual_loss(x, y):
    return nn.functional.mse_loss(vgg(x), vgg(y))

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    perceptual = perceptual_loss(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.5 * perceptual + 0.001 * kl_div

# ====================
# OPTIMIZER
# ====================
optimizer = optim.Adam(list(encoder.parameters()) + 
                       list(decoder.parameters()) + 
                       list(caption_projector.parameters()) +
                       list(latent_fusion.parameters()), lr=LEARNING_RATE)

# ====================
# REPARAMETERIZATION
# ====================
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ====================
# TRAIN LOOP
# ====================
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    caption_projector.train()
    latent_fusion.train()

    for i, (images, captions) in enumerate(dataloader):
        images = images.to(DEVICE)
        tokenized = clip.tokenize(captions, truncate=True).to(DEVICE)

        with torch.no_grad():
            caption_features = clip_model.encode_text(tokenized).float()

        caption_latents = caption_projector(caption_features)
        mu, logvar = encoder(images)
        z = reparameterize(mu, logvar)

        z_cond = latent_fusion(z, caption_latents)
        recon_images = decoder(z_cond)

        loss = vae_loss_function(recon_images, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}] Loss: {loss.item():.4f}")
            save_image(recon_images[:4], f"{CHECKPOINT_DIR}/recon_epoch{epoch+1}_batch{i}.png")

    torch.save(encoder.state_dict(), f"{CHECKPOINT_DIR}/vae_encoder_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/vae_decoder_epoch{epoch+1}.pth")
