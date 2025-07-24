import torch
from pathlib import Path

from utils.utils import create_model, load_checkpoint, save_samples, get_output_dirs
import config

# ----------------------------
# Settings
# ----------------------------
device = config.device
base_dir = Path(__file__).resolve().parent # Root folder
dirs = get_output_dirs()

# ----------------------------
# Load model
# ----------------------------
model, _ = create_model(lr=config.learning_rate, device=device)
load_checkpoint(model, optimizer=None, path=dirs['checkpoints'], device=device, training=False)

model.eval()

# ----------------------------
# Run inference
# ----------------------------
with torch.no_grad():
    # Generate sample and trajectory
    sample, trajectory = model.generate_dpm_solver(return_trajectory=True, device=device)
    # sample, trajectory = model.generate_ddim(return_trajectory=True, device=device)

    # Save output
    save_samples(sample, diffusion_trajectory=trajectory, filename=dirs["test"]/f'inference.png')
