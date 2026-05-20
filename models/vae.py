"""
Conditional VAE на эмбеддингах отзывов (768-dim XLM-R).

Бейзлайн для сравнения с WGAN-GP. Та же задача — генерация эмбеддингов
с условием по категории, но через вариационный автоэнкодер с ELBO.

Это требование курса (минимум 2 генеративные модели для Final).

Использование:
    from models.vae import CVAE, train_vae, sample_vae

    vae = CVAE(emb_dim=768, latent_dim=64, num_classes=10)
    history = train_vae(vae, dataloader, n_epochs=100, device='cuda')
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CVAE(nn.Module):
    """Conditional VAE для генерации эмбеддингов."""

    def __init__(
        self,
        emb_dim: int = 768,
        latent_dim: int = 64,
        num_classes: int = 10,
        label_emb_dim: int = 64,
        hidden: int = 512,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # Encoder: (emb, label) -> mu, log_var
        self.enc = nn.Sequential(
            nn.Linear(emb_dim + label_emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        # Decoder: (z, label) -> emb
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        le = self.label_emb(labels)
        h = self.enc(torch.cat([x, le], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        le = self.label_emb(labels)
        return self.dec(torch.cat([z, le], dim=-1))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar


def elbo_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """
    Negative ELBO = reconstruction (MSE) + beta * KL.
    beta = 1 для классического VAE, > 1 для beta-VAE (disentangling).
    """
    recon = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl


def train_vae(
    model: CVAE,
    dataloader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda",
    log_every: int = 10,
) -> dict:
    """Тренировка CVAE."""
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "recon": [], "kl": []}

    for epoch in range(n_epochs):
        losses, recons, kls = [], [], []
        for x, labels in dataloader:
            x = x.to(device).float()
            labels = labels.to(device).long()

            x_recon, mu, logvar = model(x, labels)
            loss, recon, kl = elbo_loss(x, x_recon, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            recons.append(recon.item())
            kls.append(kl.item())

        history["loss"].append(sum(losses) / len(losses))
        history["recon"].append(sum(recons) / len(recons))
        history["kl"].append(sum(kls) / len(kls))

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}  "
                f"loss={history['loss'][-1]:.4f}  "
                f"recon={history['recon'][-1]:.4f}  "
                f"kl={history['kl'][-1]:.4f}"
            )

    return history


@torch.no_grad()
def sample_vae(
    model: CVAE,
    n_per_class: int,
    num_classes: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Сэмплируем из VAE по n_per_class на класс."""
    model.eval()
    all_samples, all_labels = [], []
    for c in range(num_classes):
        z = torch.randn(n_per_class, model.latent_dim, device=device)
        labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        samples = model.decode(z, labels)
        all_samples.append(samples)
        all_labels.append(labels)
    model.train()
    return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)
