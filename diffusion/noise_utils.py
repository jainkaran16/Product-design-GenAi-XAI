import torch

def sample_loop(model, scheduler, shape, cond, device):
    """
    Sampling loop to generate images from noise using a model and scheduler.
    
    Args:
        model (nn.Module): The model to predict noise.
        scheduler (LinearNoiseScheduler): The noise scheduler to handle timesteps and noise levels.
        shape (tuple): Shape of the generated image (B, C, H, W).
        cond (torch.Tensor): Condition input for the model (e.g., text embedding, labels).
        device (torch.device): The device (CPU or CUDA) where computations should happen.
    
    Returns:
        torch.Tensor: Generated image.
    """
    # Start with random noise
    x = torch.randn(shape, device=device)
    
    # Iterate through timesteps in reverse order
    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((x.size(0),), t / scheduler.timesteps, device=device)
        
        # Predict the noise at timestep t
        pred_noise = model(x, t_tensor, cond)
        
        # Get the noise scaling factor (alpha_t) for timestep t
        alpha_t = scheduler.get_noise_level(t)
        
        # Update the image using the predicted noise
        x = (x - pred_noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        
        # Add noise back for all timesteps except t=0
        if t > 0:
            noise = torch.randn_like(x)
            beta_t = scheduler.betas[t]
            x += beta_t.sqrt() * noise
    
    return x
