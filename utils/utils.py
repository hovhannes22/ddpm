import torch

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from models import Diffuser

# ----------------------------
# Directories
# ----------------------------
def get_output_dirs(output_folder="outputs"):
    root = Path(__file__).resolve().parents[1]

    out_dir = root / output_folder
    checkpoints_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    loss_dir = out_dir / "loss"
    test_dir = out_dir / "test"

    for d in [out_dir, checkpoints_dir, samples_dir, loss_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "outputs": out_dir,
        "checkpoints": checkpoints_dir,
        "samples": samples_dir,
        "loss": loss_dir,
        "test": test_dir,
    }

# ----------------------------
# Training utils
# ----------------------------
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
    }, path)

def load_checkpoint(model, optimizer, path, device):
    folder = Path(path)
    files  = list(folder.glob("*.pth"))
    if not files:
        print("No checkpoint found.")
        return 0
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading checkpoint from {latest}")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch  = ckpt['epoch']
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch

def create_model(lr=2e-4, device='cuda'):
    model = Diffuser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer

# ----------------------------
# Visualization
# ----------------------------
def print_loss(epoch, step, total_steps, loss_total, loss_diff):
    print(
        f"Epoch {epoch} | Step {step}/{total_steps} "
        f"| Total Loss: {loss_total:.4f} "
        f"| Diffusion Loss: {loss_diff:.4f} "
    )

def save_samples(generated_image, target_image=None, diffusion_trajectory=None, target_image_noised=None, show_noised_target=False, filename="sample.png", debug_stats=False):
    def _to_img(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _print_stats(name: str, x: torch.Tensor):
        if debug_stats:
            print(f"{name:>18s}  mean {x.mean():.4f}  std {x.std():.4f}")

    B = generated_image.shape[0]
    T = len(diffusion_trajectory) if diffusion_trajectory is not None else 0

    show_target = target_image is not None
    show_noised_target = show_noised_target and target_image_noised is not None
    show_diffusion = diffusion_trajectory is not None

    step_indices = []
    if show_diffusion:
        n_steps = min(16, T)
        step_indices = np.linspace(0, T - 1, n_steps, dtype=int).tolist()

    num_columns = 1  # generated
    if show_target: num_columns += 1
    if show_noised_target: num_columns += 1
    if show_diffusion: num_columns += len(step_indices)

    fig, axes = plt.subplots(B, num_columns, figsize=(2 * num_columns, 2 * B))
    if B == 1:
        axes = np.expand_dims(axes, 0)

    for b in range(B):
        col = 0

        # Generated
        gen = generated_image[b, 0]
        gen_min, gen_max = gen.min().item(), gen.max().item()
        _print_stats("generated", gen)
        axes[b, col].imshow(_to_img(gen), cmap="gray", vmin=0, vmax=1)
        # axes[b, col].text(2, 2, f"min={gen_min:.3f}\nmax={gen_max:.3f}",
        #                   color="red", fontsize=6, va="top", ha="left",
        #                   backgroundcolor="black", alpha=0.6)
        axes[b, col].axis("off")
        if b == 0:
            axes[b, col].set_title("Generated")
        col += 1

        # Target
        if show_target:
            tgt = target_image[b, 0]
            _print_stats("target", tgt)
            axes[b, col].imshow(_to_img(tgt), cmap="gray", vmin=0, vmax=1)
            axes[b, col].axis("off")
            if b == 0:
                axes[b, col].set_title("Target")
            col += 1

        # Noised target
        if show_noised_target:
            nt = target_image_noised[b, 0]
            gen_min, gen_max = nt.min().item(), nt.max().item()
            _print_stats("noised", nt)
            axes[b, col].imshow(_to_img(nt), cmap="gray", vmin=0, vmax=1)
            # axes[b, col].text(2, 2, f"min={gen_min:.3f}\nmax={gen_max:.3f}",
            #                   color="red", fontsize=6, va="top", ha="left",
            #                   backgroundcolor="black", alpha=0.6)
            axes[b, col].axis("off")
            if b == 0:
                axes[b, col].set_title("xₜ")
            col += 1

        # Diffusion steps
        if show_diffusion:
            for t_idx in step_indices:
                xt = diffusion_trajectory[t_idx][b, 0]
                gen_min, gen_max = xt.min().item(), xt.max().item()
                _print_stats(f"xₜ[{t_idx}]", xt)
                axes[b, col].imshow(_to_img(xt), cmap="gray", vmin=0, vmax=1)
                # axes[b, col].text(2, 2, f"min={gen_min:.3f}\nmax={gen_max:.3f}",
                #                   color="red", fontsize=6, va="top", ha="left",
                #                   backgroundcolor="black", alpha=0.6)
                axes[b, col].axis("off")
                if b == 0:
                    axes[b, col].set_title(f"xₜ (t={t_idx})")
                col += 1

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    if debug_stats:
        print(f"Saved sample grid to {filename}")

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data  # skip smoothing if too short
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss(loss_history, filename='loss.png', smooth=True, window=500, use_log=False, last_n=2000):
    non_empty = [v for v in loss_history.values() if v]
    if not non_empty:
        print("No loss data to plot.")
        return

    # Prepare steps for the longest available series
    n_steps = max(len(v) for v in non_empty)
    steps = np.arange(1, n_steps + 1)

    plt.figure(figsize=(12, 6))

    for name, series in loss_history.items():
        # Skip if the list is empty or missing
        if not series:
            continue

        series = np.array(series, dtype=float)

        # Trim to last_n if needed
        if last_n is not None and len(series) > last_n:
            series = series[-last_n:]
            x = steps[-last_n:]
        else:
            x = steps[: len(series)]

        # Smooth only when we have enough points
        if smooth and len(series) >= window:
            series = moving_average(series, window)
            x = x[: len(series)]

        plt.plot(
            x,
            series,
            label=name.replace('_', ' ').title(),
            linewidth=2 if name == 'total' else 1.5
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({'Last ' + str(last_n) + ' steps' if last_n else 'All'})")
    if use_log:
        plt.yscale("log")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved loss plot to {filename}")
    plt.close()
