import torch

class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_noise_level(self, t):
        return self.alphas_cumprod[t]

    def add_noise(self, x0, noise, t):
        sqrt_alpha = self.alphas_cumprod[t].sqrt().to(x0.device)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
