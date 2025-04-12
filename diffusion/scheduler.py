import torch

class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_noise_level(self, t):
        return self.alphas_cumprod[t.cpu()].to(t.device)

    def add_noise(self, x0, noise, t):
        """
        Args:
            x0: clean latent tensor (B, C, H, W)
            noise: random noise tensor (B, C, H, W)
            t: tensor of timestep indices (B,)
        """
        # Corrected: use t.cpu() for indexing, then move result to x0.device
        alpha_t = self.alphas_cumprod[t.cpu()].sqrt().to(x0.device).view(-1, 1, 1, 1)
        one_minus_alpha_t = (1 - self.alphas_cumprod[t.cpu()]).sqrt().to(x0.device).view(-1, 1, 1, 1)

        return alpha_t * x0 + one_minus_alpha_t * noise
