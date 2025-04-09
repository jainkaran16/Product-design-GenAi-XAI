import os
import torch
import clip
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import vgg16
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils.dataset_loader import ImageCaptionDataset
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder

# ====================
# CONFIG
# ====================
LATENT_DIM = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "/content/data/Furniture Dataset"
CHECKPOINT_DIR = "/content/Product-design-GenAi-XAI/checkpoints"
RECON_DIR = os.path.join(CHECKPOINT_DIR, "recon_samples")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RECON_DIR, exist_ok=True)

# ====================
# TRANSFORM
# ====================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ====================
# DATASET
# ====================
dataset = ImageCaptionDataset(DATASET_PATH, transform=transform, use_caption=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================
# CLIP MODEL
# ====================
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ====================
# MODELS
# ====================
encoder = VAEEncoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(latent_dim=LATENT_DIM).to(DEVICE)
caption_projector = nn.Linear(512, LATENT_DIM).to(DEVICE)

class LatentFusion(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z_img, z_txt):
        return self.fusion(torch.cat([z_img, z_txt], dim=1))

latent_fusion = LatentFusion(LATENT_DIM).to(DEVICE)

# ====================
# VGG FOR PERCEPTUAL LOSS
# ====================
vgg = vgg16(pretrained=True).features[:16].to(DEVICE).eval()
for param in vgg.parameters():
    param.requires_grad = False

# ====================
# LOSS FUNCTION
# ====================
def perceptual_loss(x, y):
    return nn.functional.mse_loss(vgg(x), vgg(y))

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    perceptual = perceptual_loss(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + 1.0 * perceptual + 0.0005 * kl_div
    return total, recon_loss.item(), perceptual.item(), kl_div.item()

# ====================
# OPTIMIZER + AMP
# ====================
optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()) +
    list(caption_projector.parameters()) +
    list(latent_fusion.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
scaler = GradScaler(device_type='cuda')
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
best_loss = float('inf')

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    caption_projector.train()
    latent_fusion.train()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0

    for i, (images, captions) in enumerate(pbar):
        images = images.to(DEVICE)
        tokenized = clip.tokenize(captions, truncate=True).to(DEVICE)

        with torch.no_grad():
            caption_features = clip_model.encode_text(tokenized).float()

        caption_latents = caption_projector(caption_features)
        mu, logvar = encoder(images)
        z = reparameterize(mu, logvar)
        z_cond = latent_fusion(z, caption_latents)

        with autocast(device_type='cuda'):  # ✅ fix warning
            recon_images = decoder(z_cond)
            loss, recon_l, perc_l, kl_l = vae_loss_function(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        if i % 50 == 0:
            recon_images_save = (recon_images[:4] + 1) / 2.0
            save_image(recon_images_save, f"{RECON_DIR}/recon_epoch{epoch+1}_batch{i}.png")
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_l:.3f}",
                "Perc": f"{perc_l:.3f}",
                "KL": f"{kl_l:.5f}"
            })

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(encoder.state_dict(), f"{CHECKPOINT_DIR}/best_vae_encoder.pth")
        torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/best_vae_decoder.pth")

    torch.save(encoder.state_dict(), f"{CHECKPOINT_DIR}/vae_encoder_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"{CHECKPOINT_DIR}/vae_decoder_epoch{epoch+1}.pth")
