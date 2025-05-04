import torch
from torchvision.utils import save_image
from models.vae_decoder import VAEDecoder
from models.text_encoder import CaptionProjector, LatentProjector  # ✅ Import both
from models.unet import UNetModel
from diffusion.scheduler import LinearNoiseScheduler
from diffusion.noise_utils import sample_loop

# 📍 Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📝 Take user input
user_prompt = input("Enter your image prompt: ")
prompt = [user_prompt]

# 📦 Load model components
caption_encoder = CaptionProjector(latent_dim=128).to(device)
latent_projector = LatentProjector(latent_dim=128).to(device)  # ✅ Instantiate
unet = UNetModel().to(device)
vae_decoder = VAEDecoder(latent_dim=128).to(device)
scheduler = LinearNoiseScheduler()

# 📂 Load pre-trained weights
caption_encoder.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/best_caption_projector.pth", map_location=device))
latent_projector.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/best_latent_projector.pth", map_location=device))
unet.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/unet_final.pth", map_location=device))
vae_decoder.load_state_dict(torch.load("/content/drive/MyDrive/genai_checkpoints/vae_decoder.pth", map_location=device))

caption_encoder.eval()
latent_projector.eval()
unet.eval()
vae_decoder.eval()

# 🎨 Generate image
with torch.no_grad():
    cond = caption_encoder(prompt)  # 👈 This encodes the prompt
    fused_cond = latent_projector(cond)  # 👈 This fuses or projects to latent-compatible cond

    latents = sample_loop(unet, scheduler, shape=(1, 4, 32, 32), cond=fused_cond, device=device)
    image = vae_decoder(latents)
    save_image(image, "generated.png")

print("✅ Image saved as 'generated.png'")
