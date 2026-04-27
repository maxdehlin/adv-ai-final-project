"""
Autoencoder-based trajectory trust scoring.

The default β_AE path is unsupervised: train the autoencoder on the mixed demo
set itself, assuming clean expert behavior is the common mode. Trajectories that
are unusual under that learned reconstruction model receive lower trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .trust import robust_standardize_against_reference, trajectory_summary


@dataclass
class AutoencoderResult:
    scores: np.ndarray
    weights: np.ndarray
    train_losses: list[float]
    val_losses: list[float]
    threshold: float
    scale: float


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "Autoencoder trust scoring requires torch. Install the project "
            "dependencies from README.md, then rerun this command."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def build_autoencoder(nn, input_dim: int, latent_dim: int, hidden_dim: int):
    class TrajectoryAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(latent_dim * 2, latent_dim)),
                nn.ReLU(),
                nn.Linear(max(latent_dim * 2, latent_dim), latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, max(latent_dim * 2, latent_dim)),
                nn.ReLU(),
                nn.Linear(max(latent_dim * 2, latent_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return TrajectoryAutoencoder()


def fit_autoencoder_scores(
    reference_summaries: np.ndarray,
    candidate_summaries: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    latent_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    val_frac: float = 0.2,
    seed: int = 42,
    device_arg: Optional[str] = None,
    patience: int = 30,
) -> AutoencoderResult:
    """
    Train an autoencoder on reference summaries and score candidates.

    In the default ANTIDOTE use case, reference_summaries and
    candidate_summaries are both the mixed demo set. Larger reconstruction error
    means more outlier-like; larger weight means more trustworthy.
    """
    torch, nn, F, DataLoader, TensorDataset = import_torch()

    reference = np.asarray(reference_summaries, dtype=np.float64)
    candidates = np.asarray(candidate_summaries, dtype=np.float64)
    if reference.ndim != 2 or candidates.ndim != 2:
        raise ValueError("reference_summaries and candidate_summaries must be 2-D arrays.")
    if reference.shape[1] != candidates.shape[1]:
        raise ValueError("Reference and candidate summaries must have the same feature dimension.")
    if len(reference) < 4:
        raise ValueError("Need at least 4 reference trajectories for autoencoder training.")

    ref_scaled, cand_scaled = robust_standardize_against_reference(reference, candidates)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ref_scaled))
    ref_scaled = ref_scaled[order]

    n_val = int(round(len(ref_scaled) * val_frac))
    n_val = min(max(1, n_val), len(ref_scaled) - 1)
    val_x_np = ref_scaled[:n_val].astype(np.float32)
    train_x_np = ref_scaled[n_val:].astype(np.float32)

    input_dim = reference.shape[1]
    if latent_dim is None:
        latent_dim = max(2, min(16, input_dim // 3))
    if hidden_dim is None:
        hidden_dim = max(16, min(128, input_dim * 2))

    torch.manual_seed(seed)
    device = torch.device(device_arg if device_arg else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_autoencoder(nn, input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.tensor(train_x_np, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_x = torch.tensor(val_x_np, dtype=torch.float32, device=device)

    train_losses = []
    val_losses = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = F.mse_loss(recon, xb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_recon = model(val_x)
            val_loss = float(F.mse_loss(val_recon, val_x).detach().cpu())

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience > 0 and epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    cand_x = torch.tensor(cand_scaled.astype(np.float32), dtype=torch.float32, device=device)
    ref_x = torch.tensor(ref_scaled.astype(np.float32), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        cand_recon = model(cand_x)
        ref_recon = model(ref_x)
        scores = ((cand_recon - cand_x) ** 2).mean(dim=1).detach().cpu().numpy()
        ref_errors = ((ref_recon - ref_x) ** 2).mean(dim=1).detach().cpu().numpy()

    threshold = float(np.percentile(ref_errors, 95))
    q25 = float(np.percentile(ref_errors, 25))
    q75 = float(np.percentile(ref_errors, 75))
    scale = q75 - q25
    if scale < 1e-8:
        scale = float(np.std(ref_errors))
    if scale < 1e-8:
        scale = 1.0

    weights = reconstruction_errors_to_weights(scores, threshold=threshold, scale=scale)
    return AutoencoderResult(
        scores=scores,
        weights=weights,
        train_losses=train_losses,
        val_losses=val_losses,
        threshold=threshold,
        scale=scale,
    )


def beta_AE(
    demo_trajs,
    *,
    reference_trajs=None,
    summary_mode: str = "action",
    frame_stride: int = 10,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    latent_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    val_frac: float = 0.2,
    seed: int = 42,
    device_arg: Optional[str] = None,
    patience: int = 30,
    return_result: bool = False,
):
    """
    β_AE: Autoencoder trust weights.

    Train an autoencoder on trajectory summaries, then trust demo trajectories
    in proportion to how well they reconstruct. By default, the autoencoder is
    trained on demo_trajs themselves, which matches the majority-clean setting:
    expert behavior should be common and reconstruct better than poison.

    Returns (N_demo,) weights in [0, 1] by default. Set return_result=True to
    receive the full AutoencoderResult with reconstruction scores and losses.
    """
    if reference_trajs is None:
        reference_trajs = demo_trajs

    reference_summaries = np.array(
        [
            trajectory_summary(t.states, t.actions, mode=summary_mode, frame_stride=frame_stride)[0]
            for t in reference_trajs
        ]
    )
    demo_summaries = np.array(
        [
            trajectory_summary(t.states, t.actions, mode=summary_mode, frame_stride=frame_stride)[0]
            for t in demo_trajs
        ]
    )

    result = fit_autoencoder_scores(
        reference_summaries,
        demo_summaries,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        val_frac=val_frac,
        seed=seed,
        device_arg=device_arg,
        patience=patience,
    )
    return result if return_result else result.weights


def reconstruction_errors_to_weights(
    errors: np.ndarray,
    *,
    threshold: float,
    scale: float,
    temperature: float = 1.0,
) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    scale = float(scale) if scale > 1e-8 else 1.0
    z = (errors - float(threshold)) / scale
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(float(temperature) * z))
