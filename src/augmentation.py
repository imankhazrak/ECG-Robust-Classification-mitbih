"""
ECG beat augmentation for training: Gaussian noise, time jitter, amplitude scaling.
Applied stochastically during training to improve minority-class generalization.
"""
import random

import torch


class ECGAugment:
    """Composable ECG augmentation with configurable probabilities."""

    def __init__(
        self,
        noise_std: float = 0.05,
        jitter_ms: int = 10,
        sample_rate: int = 360,
        scale_range: tuple[float, float] = (0.95, 1.05),
        p_noise: float = 0.5,
        p_jitter: float = 0.5,
        p_scale: float = 0.5,
    ):
        """
        Args:
            noise_std: Std of Gaussian noise added (signal is z-score normalized ~[-3, 3]).
            jitter_ms: Max circular shift in ms (positive and negative).
            sample_rate: Samples per second for jitter conversion.
            scale_range: (low, high) for amplitude scaling.
            p_*: Probability of applying each augmentation.
        """
        self.noise_std = noise_std
        self.jitter_samples = int(jitter_ms * sample_rate / 1000)
        self.scale_range = scale_range
        self.p_noise = p_noise
        self.p_jitter = p_jitter
        self.p_scale = p_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to beat x: (seq_len, C)."""
        x = x.clone()
        if random.random() < self.p_noise:
            x = x + torch.randn_like(x) * self.noise_std
        if random.random() < self.p_jitter and self.jitter_samples > 0:
            shift = random.randint(-self.jitter_samples, self.jitter_samples)
            x = torch.roll(x, shift, dims=0)
        if random.random() < self.p_scale:
            s = random.uniform(*self.scale_range)
            x = x * s
        return x


def build_augment(config: dict | None) -> ECGAugment | None:
    """Build ECGAugment from config dict; return None if augment disabled."""
    if config is None or not config.get("augment", False):
        return None
    return ECGAugment(
        noise_std=config.get("noise_std", 0.05),
        jitter_ms=config.get("jitter_ms", 10),
        sample_rate=config.get("sample_rate", 360),
        scale_range=tuple(config.get("scale_range", [0.95, 1.05])),
        p_noise=config.get("p_noise", 0.5),
        p_jitter=config.get("p_jitter", 0.5),
        p_scale=config.get("p_scale", 0.5),
    )
