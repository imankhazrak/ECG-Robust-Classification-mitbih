"""
Grad-CAM visualization for CNN + Transformer ECG model.
Produces heatmap overlay on input ECG showing discriminative regions.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM heatmap for a single ECG input.
    Args:
        model: CNNTransformer model.
        x: (1, 1, 187) input tensor.
        target_class: Class to visualize; None = predicted class.
        target_layer: Layer to hook; default = last conv in CNN.
    Returns:
        heatmap: (187,) normalized heatmap.
        pred_class: Predicted class index.
    """
    model.eval()
    x = x.requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    if target_layer is None:
        target_layer = model.cnn.features[-2]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    pred_class = logits.argmax(1).item()
    c = target_class if target_class is not None else pred_class

    model.zero_grad()
    logits[0, c].backward()

    h_fwd.remove()
    h_bwd.remove()

    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=1)
    cam = (weights.unsqueeze(1) * act).sum(dim=0)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()

    act_np = act.cpu().numpy()
    if act_np.ndim == 1:
        heatmap = cam
    else:
        h_len = cam.shape[-1]
        heatmap = np.interp(
            np.linspace(0, h_len - 1, x.shape[-1]),
            np.arange(h_len),
            cam,
        )

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, pred_class


def plot_gradcam(
    x: np.ndarray,
    heatmap: np.ndarray,
    pred_class: int,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot ECG with Grad-CAM heatmap overlay."""
    x = x.squeeze()
    t = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x, "b-", label="ECG", linewidth=1)
    ax.fill_between(t, x.min(), x, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, heatmap, "r-", alpha=0.7, label="Grad-CAM")
    ax2.set_ylabel("Activation")
    ax2.set_ylim(0, 1.1)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Predicted: {class_names[pred_class]}")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
