import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CaptionProjector(nn.Module):
    def __init__(self, device='cuda', latent_dim=512):
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.projection = nn.Linear(512, latent_dim)  # Project to latent dim of VAE

    def forward(self, captions):
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        return self.projection(text_features)
