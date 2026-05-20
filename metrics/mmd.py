"""
Maximum Mean Discrepancy (MMD) — статистическое сравнение двух распределений
в пространстве эмбеддингов.

MMD = 0 если распределения одинаковые, > 0 иначе.

Используется как основная метрика близости сгенерированных и реальных эмбеддингов.
Не требует обученной модели-классификатора, в отличие от FID.

Reference: Gretton et al., "A Kernel Two-Sample Test" (2012).
"""
from __future__ import annotations

import numpy as np
import torch


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: list[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Mixture of Gaussian RBF kernels с разными ширинами."""
    # x: (N, D), y: (M, D)
    xx = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    yy = (y * y).sum(dim=1, keepdim=True)  # (M, 1)
    xy = x @ y.T  # (N, M)
    dist2 = xx + yy.T - 2 * xy
    dist2 = dist2.clamp(min=0.0)

    k = torch.zeros_like(dist2)
    for s in sigmas:
        k = k + torch.exp(-dist2 / (2 * s * s))
    return k / len(sigmas)


def mmd2_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: list[float] | None = None,
) -> float:
    """
    Unbiased MMD^2 estimator.
    Меньше = распределения ближе.
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 4.0, 8.0, 16.0]

    n = x.size(0)
    m = y.size(0)

    k_xx = gaussian_kernel(x, x, sigmas)
    k_yy = gaussian_kernel(y, y, sigmas)
    k_xy = gaussian_kernel(x, y, sigmas)

    # Убираем диагональ для unbiased (Gretton 2012)
    k_xx = k_xx - torch.diag(torch.diag(k_xx))
    k_yy = k_yy - torch.diag(torch.diag(k_yy))

    term_xx = k_xx.sum() / (n * (n - 1))
    term_yy = k_yy.sum() / (m * (m - 1))
    term_xy = k_xy.sum() / (n * m)

    return float((term_xx + term_yy - 2 * term_xy).item())


def compute_mmd(
    real: np.ndarray | torch.Tensor,
    fake: np.ndarray | torch.Tensor,
    sigmas: list[float] | None = None,
    device: str = "cpu",
    max_samples: int = 2000,
) -> float:
    """
    Удобная обёртка: принимает numpy/torch, возвращает float.
    Сабсемплирует до max_samples для скорости (MMD O(N^2)).
    """
    if isinstance(real, np.ndarray):
        real = torch.from_numpy(real).float()
    if isinstance(fake, np.ndarray):
        fake = torch.from_numpy(fake).float()

    if real.size(0) > max_samples:
        idx = torch.randperm(real.size(0))[:max_samples]
        real = real[idx]
    if fake.size(0) > max_samples:
        idx = torch.randperm(fake.size(0))[:max_samples]
        fake = fake[idx]

    real = real.to(device)
    fake = fake.to(device)
    return mmd2_unbiased(real, fake, sigmas)


def compute_mmd_per_class(
    real: np.ndarray,
    fake: np.ndarray,
    real_labels: np.ndarray,
    fake_labels: np.ndarray,
    num_classes: int,
) -> dict[int, float]:
    """MMD отдельно для каждого класса — увидим где модель сильнее/слабее."""
    result = {}
    for c in range(num_classes):
        r = real[real_labels == c]
        f = fake[fake_labels == c]
        if len(r) < 10 or len(f) < 10:
            result[c] = float("nan")
            continue
        result[c] = compute_mmd(r, f)
    return result
