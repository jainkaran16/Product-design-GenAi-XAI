import torch

def sample_loop(model, scheduler, shape, cond, device):
    x = torch.randn(shape).to(device)
    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.tensor([t / scheduler.timesteps], device=device).repeat(x.size(0))
        pred_noise = model(x, t_tensor, cond)
        alpha = scheduler.get_noise_level(t)
        x = (x - pred_noise * (1 - alpha).sqrt()) / alpha.sqrt()
        if t > 0:
            noise = torch.randn_like(x)
            beta = scheduler.betas[t]
            x += (beta.sqrt() * noise)
    return x
