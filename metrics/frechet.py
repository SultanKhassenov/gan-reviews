"""
Frechet Distance — аналог FID, но в пространстве эмбеддингов.

Классический FID считает Frechet distance между Gaussian-аппроксимациями
двух наборов признаков (обычно из Inception). У нас данные уже в семантическом
пространстве XLM-R, поэтому считаем Frechet distance напрямую — это валидно
и встречается в литературе как "Frechet Embedding Distance".

FED = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})

Reference: Heusel et al., "GANs Trained by a TTUR..." (2017).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg


def compute_frechet_distance(
    real: np.ndarray,
    fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Frechet Distance между двумя наборами признаков.
    real, fake: (N, D)
    Меньше = распределения ближе. 0 = идентичны.
    """
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("ожидаются 2D массивы (N, D)")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("размерность признаков должна совпадать")

    mu_r = real.mean(axis=0)
    mu_g = fake.mean(axis=0)
    sigma_r = np.cov(real, rowvar=False)
    sigma_g = np.cov(fake, rowvar=False)

    diff = mu_r - mu_g

    # Численная стабилизация: добавляем eps*I перед sqrtm
    offset = np.eye(sigma_r.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sigma_r + offset) @ (sigma_g + offset), disp=False)

    # Если получились мнимые компоненты (численная погрешность) — берём вещественную часть
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"[frechet] WARN: имагинарная компонента = {m:.3e}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2 * tr_covmean)


def compute_frechet_per_class(
    real: np.ndarray,
    fake: np.ndarray,
    real_labels: np.ndarray,
    fake_labels: np.ndarray,
    num_classes: int,
) -> dict[int, float]:
    """Frechet distance отдельно для каждого класса."""
    result = {}
    for c in range(num_classes):
        r = real[real_labels == c]
        f = fake[fake_labels == c]
        if len(r) < 10 or len(f) < 10:
            result[c] = float("nan")
            continue
        result[c] = compute_frechet_distance(r, f)
    return result
