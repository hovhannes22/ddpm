import torch
import torch.nn as nn

import math

import config

# --------------------------
# ResBlock
# --------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=None):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if t_emb_dim is not None:
            self.t_emb_proj = nn.Linear(t_emb_dim, out_channels, bias=True)

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU()

        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if t_emb is not None:
            emb = self.act(self.t_emb_proj(t_emb))[:, :, None, None]
            h = h + emb

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return self.shortcut(x) + h

# --------------------------
# Timestep embedding
# --------------------------
def get_timestep_embedding(timestep, embedding_dim=256):
    timestep = timestep.float().unsqueeze(1) # (B, 1)
    
    half_dim = embedding_dim // 2
    
    # Spacing
    exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=timestep.dtype, device=timestep.device) / (half_dim - 1) # (half_dim)

    frequencies = torch.exp(exponent).unsqueeze(0) # (1, half_dim)

    # Angles
    angles = timestep * frequencies # (B, half_dim)

    # sin and cos
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # (B, embedding_dim)

    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, t_emb):
        # t_emb: (B, embedding_dim)
        out = self.block(t_emb) # (B, embedding_dim)
        return out

# --------------------------
# UNet
# --------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, attention=False):
        super().__init__()
        self.resnet = ResBlock(in_channels, out_channels, t_emb_dim)
        self.norm = nn.GroupNorm(min(32, out_channels), num_channels=out_channels)
        
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True) if attention else None

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb):
        # x: (B, in_channels, H, W)

        out = self.resnet(x, t_emb) # (B, out_channels, H, W)
        out = self.norm(out) # (B, out_channels, H, W

        residual = out
        if self.self_attention is not None:
            B, C, H, W = out.shape
            attention_out = out.view(B, C, -1).transpose(-2, -1) # (B, H*W, out_channels)
            attention_out, _ = self.self_attention(attention_out, attention_out, attention_out) # (B, H*W, out_channels)
            attention_out = attention_out.transpose(-2, -1).view(B, C, H, W) # (B, out_channels, H, W)

            residual += attention_out # (B, out_channels, H, W)

        out = self.downsample(residual) # (B, out_channels, H/2, W/2)

        return out, residual
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, t_emb_dim, attention=False):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.resnet = ResBlock(out_channels + skip_channels, out_channels, t_emb_dim)
        self.norm = nn.GroupNorm(min(32, out_channels), num_channels=out_channels)

        self.self_attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)  if attention else None

    def forward(self, x, skip, t_emb):
        # x: (B, in_channels, H, W)
        # skip: (B, out_channels, H, W)
        out = self.upsample(x) # (B, out_channels, H*2, W*2)

        out = torch.cat([out, skip], dim=1) # (B, out_channels, H*2, W*2)

        out = self.resnet(out, t_emb) # (B, out_channels, H*2, W*2)
        out = self.norm(out) # (B, out_channels, H*2, W*2)

        if self.self_attention is not None:
            B, C, H, W = out.shape
            attention_out = out.view(B, C, -1).transpose(-2, -1) # (B, H*2*W*2, out_channels)
            attention_out, _ = self.self_attention(attention_out, attention_out, attention_out) # (B, H*2*W*2, out_channels)
            attention_out = attention_out.transpose(-2, -1).view(B, C, H, W) # (B, out_channels, H*2, W*2)

            out += attention_out

        return out
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        
        self.resnet1 = ResBlock(in_channels, out_channels, t_emb_dim)

        self.norm = nn.GroupNorm(min(32, out_channels), num_channels=out_channels)
        self.self_attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)

        self.resnet2 = ResBlock(out_channels, out_channels, t_emb_dim)

    def forward(self, x, t_emb):
        # x: (B, in_channels, H, W)

        out = self.resnet1(x, t_emb) # (B, out_channels, H, W)

        B, C, H, W = out.shape
        out = self.norm(out) # (B, out_channels, H, W)
        attention_out = out.view(B, C, -1).transpose(-2, -1) # (B, H*W, out_channels)
        attention_out, _ = self.self_attention(attention_out, attention_out, attention_out) # (B, H*W, out_channels)
        attention_out = attention_out.transpose(-2, -1).view(B, C, H, W) # (B, out_channels, H, W)

        resnet_input = out + attention_out # (B, out_channels, H, W)

        out = self.resnet2(resnet_input, t_emb) # (B, out_channels, H, W)

        out = out + resnet_input # (B, out_channels, H, W)

        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, t_emb_dim, base_channels=64):
        super().__init__()

        self.t_emb_dim = t_emb_dim
        self.timestep_embedding = TimestepEmbedding(t_emb_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, t_emb_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, t_emb_dim)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, t_emb_dim, attention=True)

        self.bottleneck = MidBlock(base_channels * 4, base_channels * 4, t_emb_dim)

        self.up3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, t_emb_dim, attention=True)
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, t_emb_dim)
        self.up1 = UpBlock(base_channels, base_channels, base_channels, t_emb_dim)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(32, base_channels), num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x_t, t):
        # x_t: (B, in_channels, H, W) Noisy input image at diffusion step t
        # label: (B, vocab_szie)
        # t: (B, )

        t_emb = get_timestep_embedding(t, embedding_dim=self.t_emb_dim) # (B, t_emb_dim)
        t_emb = self.timestep_embedding(t_emb) # (B, t_emb_dim)

        x0 = self.init_conv(x_t) # (B, base_channels, H, W)
        
        x1, x1_residual = self.down1(x0, t_emb) # (B, base_channels, H/2, W/2)
        x2, x2_residual = self.down2(x1, t_emb) # (B, base_channels * 2, H/4, W/4)
        x3, x3_residual = self.down3(x2, t_emb) # (B, base_channels * 4, H/8, W/8)

        bottleneck = self.bottleneck(x3, t_emb) # (B, base_channels * 4, H/8, W/8)

        u3 = self.up3(bottleneck, x3_residual, t_emb) # (B, base_channels * 2, H/4, W/4)
        u2 = self.up2(u3, x2_residual,  t_emb) # (B, base_channels, H/2, W/2)
        u1 = self.up1(u2, x1_residual, t_emb) # (B, base_channels, H, W)

        image = self.final_conv(u1) # (B, in_channels, H, W)

        return image
    
# --------------------------
# Diffusion Generator
# --------------------------
class DiffusionGenerator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        self.t_emb_dim = 512

        # UNet
        self.unet = UNet(in_channels, self.t_emb_dim, base_channels)

    def forward(self, x_t, t):
        # x_t: (B, C, H, W) Target image with added noise
        # t: (B, 1)

        return self.unet(x_t, t)

# --------------------------
# Diffusion Model
# --------------------------
class Diffuser(nn.Module):
    def __init__(self, timesteps=1000):
        super().__init__()

        self.timesteps = timesteps

        self.diffusion_generator = DiffusionGenerator()

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.loss_function_diffusion_fn = nn.MSELoss()

    def q_sample(self, x_0, t):
        # x_0: (B, C, H, W) Initial image

        noise = torch.randn_like(x_0) # (B, C, H, W)

        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1) # (B, 1, 1, 1)
        
        x_t = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * noise # (B, C, H, W)

        return x_t, noise

    def forward(self, target_image, t):
        # target_image: (B, 1, H, W)
        # t: (B, )

        x_t, true_noise = self.q_sample(target_image, t)

        pred_noise = self.diffusion_generator(x_t, t)

        # Reconstruct x_0 from predicted noise, per sample
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt()

        x0_reconstructed = (x_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha

        loss_diff = self.loss_function_diffusion_fn(pred_noise, true_noise)

        return {
            "loss_diffusion": loss_diff,
            "generated_image": x0_reconstructed.detach(),
            "target_noised": x_t
        }
    
    @torch.no_grad()
    def generate_ddim(self, num_steps=50, eta=0.0, return_trajectory=False, device = 'cuda'):
        self.eval()
        
        B = 4
        H = config.image_size
        W = H

        # Generate noise
        x_t = torch.randn((B, 1, H, W), device=device)

        # Create timesteps
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        trajectory = [x_t] # Pure noise added to the list
        for i in range(num_steps):
            # Current and previous timestep indices
            t = timesteps[i].repeat(B)
            t_prev = timesteps[i+1].repeat(B) if i < num_steps - 1 else torch.zeros_like(t)

            # Alphas
            alpha_bar = self.alphas_cumprod[t].view(B, 1, 1, 1)
            alpha_bar_prev = self.alphas_cumprod[t_prev].view(B, 1, 1, 1)

            # Predict noise
            pred_noise = self.diffusion_generator(x_t, t)

            # Reconstruct x0
            x0_pred = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt() * (1 - alpha_bar / alpha_bar_prev).sqrt()

            if eta == 0:
                x_t = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * pred_noise
            else:
                noise = torch.randn_like(x_t)
                x_t = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev - sigma**2).sqrt() * pred_noise + sigma * noise

            if return_trajectory:
                trajectory.append(x_t.clone())

        self.train()
        if return_trajectory:
            return x_t, trajectory
        return x_t

    @torch.no_grad()
    def generate_dpm_solver(self, num_steps=20, return_trajectory=False, device='cuda'):
        self.eval()

        B = 4
        H = config.image_size
        W = H

        # Generate noise
        x_t = torch.randn(B, 1, H, W, device=device)

        # Create timesteps
        timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)

        # Helper: interpolate alphā(t) from training schedule
        def get_alpha_bar(t_frac_batch):  # shape (B,)
            t_idx = t_frac_batch * (self.timesteps - 1)
            low = t_idx.floor().long().clamp(0, self.timesteps - 2)
            high = low + 1
            w = (t_idx - low.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)

            alpha_low = self.alphas_cumprod[low].view(-1, 1, 1, 1)
            alpha_high = self.alphas_cumprod[high].view(-1, 1, 1, 1)

            return (1 - w) * alpha_low + w * alpha_high  # (B,1,1,1)

        trajectory = [x_t]

        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < num_steps else torch.tensor(0.0, device=device)

            t_index_cur = (t_cur * (self.timesteps - 1)).long()
            t_batch = t_index_cur.repeat(B)

            eps1 = self.diffusion_generator(x_t, t_batch)

            alpha_cur = get_alpha_bar(t_cur).to(device)
            x0_pred = (x_t - (1 - alpha_cur).sqrt() * eps1) / alpha_cur.sqrt()

            if i < num_steps - 1:
                alpha_next = get_alpha_bar(t_next).to(device)
                x_t_next = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * eps1
            else:
                x_t_next = x0_pred

            x_t = x_t_next
            if return_trajectory:
                trajectory.append(x_t.clone())

        self.train()
        return x_t, trajectory
    