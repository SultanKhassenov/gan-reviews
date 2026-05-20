"""
WGAN-GP для генерации эмбеддингов отзывов (768-dim XLM-R векторов).

Conditional на категории товара. Архитектура:
- Generator: MLP, z + label_embedding -> 768-dim vector
- Discriminator (Critic): MLP, vector + label_embedding -> scalar
- Loss: Wasserstein с gradient penalty (Gulrajani et al., 2017)

Использование:
    from models.wgan_gp import Generator, Critic, train_wgan_gp

    gen = Generator(z_dim=128, emb_dim=768, num_classes=10)
    crit = Critic(emb_dim=768, num_classes=10)
    history = train_wgan_gp(gen, crit, dataloader, n_epochs=200, device='cuda')
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ────────────────────────────── Архитектура ──────────────────────────────


class Generator(nn.Module):
    """
    Conditional Generator: (noise, label) -> embedding vector.

    Args:
        z_dim: размерность шума
        emb_dim: размерность выходного эмбеддинга (768 для XLM-R-base)
        num_classes: количество категорий (условий)
        label_emb_dim: размерность label embedding
        hidden: ширина скрытых слоёв
    """

    def __init__(
        self,
        z_dim: int = 128,
        emb_dim: int = 768,
        num_classes: int = 10,
        label_emb_dim: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        in_dim = z_dim + label_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim), labels: (B,) int
        label_e = self.label_emb(labels)  # (B, label_emb_dim)
        x = torch.cat([z, label_e], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    """
    Conditional Critic (Discriminator без sigmoid для WGAN).

    Принимает (embedding, label) и выдаёт scalar — оценку Wasserstein.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_classes: int = 10,
        label_emb_dim: int = 64,
        hidden: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        in_dim = emb_dim + label_emb_dim
        # В WGAN-GP **нельзя** использовать BatchNorm в критике
        # (нарушает Lipschitz). Используем LayerNorm или ничего.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_e = self.label_emb(labels)
        inp = torch.cat([x, label_e], dim=-1)
        return self.net(inp).squeeze(-1)


# ────────────────────────────── Потери ──────────────────────────────


def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Gradient penalty (Gulrajani et al., 2017)."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    scores = critic(interpolated, labels)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# ────────────────────────────── Тренировка ──────────────────────────────


def train_wgan_gp(
    generator: Generator,
    critic: Critic,
    dataloader: DataLoader,
    n_epochs: int = 200,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.5, 0.9),
    device: str = "cuda",
    log_every: int = 10,
) -> dict:
    """
    Тренировка WGAN-GP.

    Args:
        dataloader: даёт батчи (embeddings, labels) — (B, 768), (B,)
        n_critic: сколько шагов критика на 1 шаг генератора
        lambda_gp: коэффициент gradient penalty
        betas: для Adam (рекомендация Gulrajani — (0.5, 0.9))

    Returns:
        history: {'d_loss': [...], 'g_loss': [...], 'gp': [...]}
    """
    generator.to(device)
    critic.to(device)
    generator.train()
    critic.train()

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    history = {"d_loss": [], "g_loss": [], "gp": []}

    for epoch in range(n_epochs):
        d_losses, g_losses, gps = [], [], []

        for step, (real, labels) in enumerate(dataloader):
            real = real.to(device).float()
            labels = labels.to(device).long()
            bs = real.size(0)

            # ── Critic step ────────────────────────────────
            for _ in range(n_critic):
                z = torch.randn(bs, generator.z_dim, device=device)
                fake = generator(z, labels).detach()

                d_real = critic(real, labels).mean()
                d_fake = critic(fake, labels).mean()
                gp = gradient_penalty(critic, real, fake, labels, device)

                d_loss = -(d_real - d_fake) + lambda_gp * gp

                opt_c.zero_grad()
                d_loss.backward()
                opt_c.step()

            d_losses.append(d_loss.item())
            gps.append(gp.item())

            # ── Generator step ─────────────────────────────
            z = torch.randn(bs, generator.z_dim, device=device)
            fake = generator(z, labels)
            g_loss = -critic(fake, labels).mean()

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())

        # Эпоха закончилась — усредняем
        d_avg = sum(d_losses) / max(1, len(d_losses))
        g_avg = sum(g_losses) / max(1, len(g_losses))
        gp_avg = sum(gps) / max(1, len(gps))
        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)
        history["gp"].append(gp_avg)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}  "
                f"D_loss={d_avg:.4f}  G_loss={g_avg:.4f}  GP={gp_avg:.4f}"
            )

    return history


# ────────────────────────────── Sampling ──────────────────────────────


@torch.no_grad()
def sample(
    generator: Generator,
    n_per_class: int,
    num_classes: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Сэмплим по n_per_class векторов на каждый класс.

    Returns:
        samples: (num_classes * n_per_class, emb_dim)
        labels: (num_classes * n_per_class,)
    """
    generator.eval()
    all_samples, all_labels = [], []
    for c in range(num_classes):
        z = torch.randn(n_per_class, generator.z_dim, device=device)
        labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        samples = generator(z, labels)
        all_samples.append(samples)
        all_labels.append(labels)
    generator.train()
    return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)
