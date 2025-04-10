import torch
from torchvision.utils import save_image
from models.vae_decoder import VAEDecoder
from models.text_encoder import CaptionProjector
from models.unet import UNetModel
from diffusion.scheduler import LinearNoiseScheduler
from diffusion.noise_utils import sample_loop

prompt = ["a surreal painting of a cyberpunk city"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

caption_encoder = CaptionProjector().to(device)
unet = UNetModel().to(device)
vae_decoder = VAEDecoder().to(device)
scheduler = LinearNoiseScheduler()

# Load trained weights
caption_encoder.eval()
unet.load_state_dict(torch.load("weights/unet.pth"))
unet.eval()
vae_decoder.eval()

with torch.no_grad():
    cond = caption_encoder(prompt)
    latents = sample_loop(unet, scheduler, shape=(1, 4, 32, 32), cond=cond, device=device)
    image = vae_decoder(latents)
    save_image(image, "output/generated.png")
