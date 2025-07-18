import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ERKATAGIR
from utils.utils import get_output_dirs, print_loss, plot_loss, save_samples, create_model, save_checkpoint, load_checkpoint

from pathlib import Path

import config

# ----------------------------
# Parameters
# ----------------------------
epochs = config.epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
T_timestep = config.T_timestep

save_every = config.save_every
sample_every = config.sample_every

device = config.device

base_dir = Path(__file__).resolve().parent # Root folder
dirs = get_output_dirs()

# ----------------------------
# Training
# ----------------------------

def train_model():
    # Create model
    model, optimizer = create_model(learning_rate, device)
    
    # Load if checkpoints available
    start_epoch = load_checkpoint(model, optimizer, dirs['checkpoints'], device)

    # Create dataset
    dataset = ERKATAGIR('./fonts')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = {
        'diffusion': [],
    }

    for epoch in range(start_epoch, epochs):
        model.train()
        total_steps = len(dataloader)
        for i, batch in enumerate(dataloader):
            target_image = batch['target_image'].to(device) # (B, 1, H, W)

            t = torch.randint(0, T_timestep, (batch_size,), device=device)  # (B, )

            # Forward
            output = model(target_image, t)

            generated_image = output["generated_image"]
            target_noised = output["target_noised"]
            loss_diff = output["loss_diffusion"]

            # Loss
            loss = loss_diff

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print
            print_loss(epoch + 1, i + 1, total_steps, loss.item(), loss_diff.item())
            losses['diffusion'].append(loss_diff.item())

            if i % sample_every == 0:
                with torch.no_grad():
                    # Generate full denoising trajectory
                    test_generated_image, trajectory = model.generate_dpm_solver(return_trajectory=True)
                    save_samples(test_generated_image, diffusion_trajectory=trajectory, filename=dirs["test"]/f'test_{epoch}_{i}.png')

                save_samples(generated_image, target_image=target_image, target_image_noised=target_noised, show_noised_target=True, filename=dirs["samples"] / f"train_{epoch}_{i}.png")
                plot_loss(losses, smooth=True, window=200, filename=dirs["loss"]/f'loss_{epoch}_{i}.png') # Plot loss

            # Save checkpoint every N steps
            if i % save_every == 0 and i != 0:
                save_checkpoint(model, optimizer, epoch, dirs["checkpoints"]/f'state_{epoch}_{i}.pth')
                print(f"Saved checkpoint at epoch {epoch}")
            
    save_checkpoint(model, optimizer, epoch, dirs["checkpoints"]/f'state_{epoch}_{i}.pth')
            
train_model()