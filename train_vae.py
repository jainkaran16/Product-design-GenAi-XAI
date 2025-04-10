import os
import math
import torch
import clip
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import vgg16
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

from utils.dataset_loader import ImageCaptionDataset
from models.vae_encoder import VAEEncoder
from models.vae_decoder import VAEDecoder
from pytorch_msssim import ssim

# ====================
# CONFIG
# ====================
LATENT_DIM = 128
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
OUT_RES = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "/content/data/Furniture Dataset"
CHECKPOINT_DIR = "/content/Product-design-GenAi-XAI/checkpoints"
RECON_DIR = os.path.join(CHECKPOINT_DIR, "recon_samples")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RECON_DIR, exist_ok=True)

config = {
    "recon_loss_weight": 1.0,
    "perceptual_loss_weight": 0.3,
    "kl_loss_weight": 1e-6,
    "kl_anneal_start": 5,
    "kl_anneal_end": 30,
    "ssim_loss_weight": 0.2,
}

# ====================
# TRANSFORMS
# ====================
transform = transforms.Compose([
    transforms.Resize((OUT_RES, OUT_RES)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
# MODEL DEFINITIONS
# ====================
encoder = VAEEncoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder = VAEDecoder(latent_dim=LATENT_DIM, out_res=OUT_RES).to(DEVICE)
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
# VGG16 Perceptual Loss
# ====================
vgg = vgg16(weights=None).features[:16].to(DEVICE).eval()
vgg.load_state_dict(torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:16].state_dict())
for param in vgg.parameters():
    param.requires_grad = False

def compute_perceptual_loss(x, y):
    return F.mse_loss(vgg(x), vgg(y))

def compute_ssim_loss(x, y):
    x = (x + 1) / 2.0
    y = (y + 1) / 2.0
    with torch.no_grad():
        ssim_val = ssim(x, y, data_range=1.0, size_average=True)
        return 1 - (ssim_val if not torch.isnan(ssim_val) else torch.tensor(0.0, device=x.device))

def get_kl_weight(epoch, config):
    start, end = config["kl_anneal_start"], config["kl_anneal_end"]
    if epoch < start: return 0.0
    elif epoch > end: return config["kl_loss_weight"]
    t = (epoch - start) / (end - start)
    return config["kl_loss_weight"] * (1 / (1 + math.exp(-12 * (t - 0.5))))

def vae_loss_function(recon_x, x, mu, logvar, epoch, config):
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    perc_loss = compute_perceptual_loss(recon_x, x)
    ssim_loss = compute_ssim_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    kl_weight = get_kl_weight(epoch, config)

    total_loss = (
        config["recon_loss_weight"] * recon_loss +
        config["perceptual_loss_weight"] * perc_loss +
        config["ssim_loss_weight"] * ssim_loss +
        kl_weight * kl_loss
    )

    return total_loss, recon_loss.item(), perc_loss.item(), ssim_loss.item(), kl_loss.item()

def kl_annealing(epoch, start=5, end=30):
    if epoch < start:
        return 0.0
    elif epoch > end:
        return 1.0
    return (epoch - start) / (end - start)

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
scaler = GradScaler()

def reparameterize(mu, logvar):
    logvar = torch.clamp(logvar, min=-6.0, max=2.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch.clamp(mu + eps * std, -6.0, 6.0)

# ====================
# TRAINING LOOP
# ====================
best_loss = float('inf')

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    caption_projector.train()
    latent_fusion.train()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    base_kl_weight = 1e-6
    kl_weight = kl_annealing(epoch, start=5, end=30) * base_kl_weight  # 👈 insert this inside loop
    for i, (images, captions) in enumerate(pbar):
        try:
            images = images.to(DEVICE)
            tokenized = clip.tokenize(captions, truncate=True).to(DEVICE)
            with torch.no_grad():
                caption_features = clip_model.encode_text(tokenized).float()
            caption_latents = caption_projector(caption_features)

            mu, logvar, skip_connections = encoder(images)
            mu = torch.tanh(mu) * 5
            logvar = torch.tanh(logvar) * 2  # so they don’t blow up
            z = reparameterize(mu, logvar)
            z_cond = latent_fusion(z, caption_latents)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                recon_images = decoder(z_cond, skip_connections)
                loss, recon_l, perc_l, ssim_l, kl_l = vae_loss_function(
                    recon_images, images, mu, logvar, epoch, config
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 50 == 0:
                grid = torch.cat([(images[:4] + 1) / 2.0, (recon_images[:4] + 1) / 2.0], dim=0)
                save_image(grid, f"{RECON_DIR}/recon_epoch{epoch+1}_batch{i}.png", nrow=4)

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Recon": f"{recon_l:.3f}",
                    "Perc": f"{perc_l:.3f}",
                    "SSIM": f"{ssim_l:.3f}",
                    "KL": f"{kl_l:.5f}"
                })

        except Exception as e:
            print(f"[Warning] Skipping corrupted batch: {e}")
            continue

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, "best_vae_encoder.pth"))
        torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, "best_vae_decoder.pth"))

    torch.save(encoder.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_encoder_epoch{epoch+1}.pth"))
    torch.save(decoder.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_decoder_epoch{epoch+1}.pth"))

    if epoch == 2:
        z_random = torch.randn(4, LATENT_DIM).to(DEVICE)
        skip = [None] * 5
        with torch.no_grad():
            rand_gen = decoder(z_random, skip)
            save_image((rand_gen + 1) / 2.0, f"{RECON_DIR}/random_gen_epoch{epoch+1}.png", nrow=2)
