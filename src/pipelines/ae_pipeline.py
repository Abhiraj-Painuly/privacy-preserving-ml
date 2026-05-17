"""Autoencoder pipeline (paper Fig 1a, §III-A).

Workflow:
    perturbed features -> AE (Vanilla / DP-VAE / TabDAE) -> latent z
        -> [z || non-sensitive] -> downstream classifier (logistic regression)
        -> evaluate (accuracy, MIA AUC, GAP MIA AUC)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..attacks.mia_aggregated import aggregated_mia_auc
from ..attacks.mia_gap import gap_mia_auc
from ..models.autoencoders import DPVAE, TabDAE, VanillaAE
from ..privacy.dp_sgd import make_private, spent_epsilon
from ..utils import device, set_seed
from ._common import DPMechanismName, prepare

AEName = Literal["vanilla_ae", "dp_vae", "tab_dae"]


@dataclass
class AEPipelineResult:
    ae: AEName
    epsilon: float
    mechanism: DPMechanismName
    accuracy: float
    mia_aggregated_auc: float
    mia_gap_auc: float
    realised_epsilon: float
    n_train: int
    n_test: int


def _build_ae(name: AEName, input_dim: int) -> nn.Module:
    if name == "vanilla_ae":
        return VanillaAE(input_dim=input_dim)
    if name == "dp_vae":
        return DPVAE(input_dim=input_dim)
    if name == "tab_dae":
        return TabDAE(input_dim=input_dim)
    raise ValueError(f"Unknown AE name: {name}")


def _train_ae(
    name: AEName,
    X_train: np.ndarray,
    *,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    batch_size: int,
    use_dpsgd: bool,
) -> tuple[nn.Module, float]:
    """Train one of the three AEs and return ``(trained_model, realised_epsilon)``."""
    dev = device()
    model = _build_ae(name, X_train.shape[1]).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    realised_eps = float(target_epsilon)
    if use_dpsgd:
        state, model, optimizer, loader = make_private(
            model,
            optimizer,
            loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
        )
    else:
        state = None  # type: ignore

    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(dev)
            optimizer.zero_grad()
            out = model(xb)
            if name == "dp_vae":
                # DPVAE.forward returns dict; recover μ, logvar, x_hat.
                x_hat, mu, logvar = out["x_hat"], out["mu"], out["logvar"]
                recon = torch.mean((x_hat - xb) ** 2, dim=-1).mean()
                kl = (-0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
                )).mean()
                loss = recon + kl
            else:
                # Vanilla AE / TabDAE: forward returns (x_hat, z).
                x_hat, _ = out
                loss = torch.mean((x_hat - xb) ** 2)
            loss.backward()
            optimizer.step()

    if use_dpsgd and state is not None:
        realised_eps = spent_epsilon(state)
    return model, realised_eps


@torch.no_grad()
def _encode(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return latent representations for every row of ``X``.

    Works whether ``model`` is the raw nn.Module or an Opacus
    ``GradSampleModule`` wrapper (which proxies through ``_module``).
    """
    model.eval()
    dev = device()
    xb = torch.from_numpy(X.astype(np.float32)).to(dev)

    # Unwrap Opacus to reach the original module that exposes ``.encode()``.
    inner = getattr(model, "_module", model)
    if hasattr(inner, "encode"):
        z = inner.encode(xb)
        if isinstance(z, tuple):
            z = z[0]  # DP-VAE.encode returns (mu, logvar)
        return z.cpu().numpy()

    # Fallback: rerun the full forward and pull z from the output.
    out = model(xb)
    if isinstance(out, tuple):
        return out[1].cpu().numpy()
    if isinstance(out, dict):
        return out.get("mu", out.get("z")).cpu().numpy()
    raise RuntimeError("Cannot derive latent z from model output type")


def run_ae_pipeline(
    df: pd.DataFrame,
    *,
    target: str,
    declared_sensitive: list[str],
    ae_name: AEName,
    epsilon: float,
    mechanism: DPMechanismName = "laplace",
    delta: float = 1e-5,
    use_pca: bool = False,
    epochs: int = 10,
    batch_size: int = 128,
    use_dpsgd: bool = True,
    seed: int = 42,
) -> AEPipelineResult:
    set_seed(seed)
    prepared = prepare(
        df,
        target=target,
        declared_sensitive=declared_sensitive,
        epsilon=epsilon,
        mechanism=mechanism,
        delta=delta,
        use_pca=use_pca,
        seed=seed,
    )

    model, realised_eps = _train_ae(
        ae_name,
        prepared.X_train,
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=epochs,
        batch_size=batch_size,
        use_dpsgd=use_dpsgd,
    )

    z_train = _encode(model, prepared.X_train)
    z_test = _encode(model, prepared.X_test)

    # Concatenate latent with non-sensitive original features (paper Fig 1a "merge").
    if prepared.non_sensitive_idx:
        ns_train = prepared.X_train[:, prepared.non_sensitive_idx]
        ns_test = prepared.X_test[:, prepared.non_sensitive_idx]
        feat_train = np.concatenate([z_train, ns_train], axis=1)
        feat_test = np.concatenate([z_test, ns_test], axis=1)
    else:
        feat_train, feat_test = z_train, z_test

    clf = LogisticRegression(max_iter=2000)
    clf.fit(feat_train, prepared.y_train)
    y_pred = clf.predict(feat_test)
    proba_test = clf.predict_proba(feat_test)
    proba_train = clf.predict_proba(feat_train)

    # Member / non-member sets for MIA: train rows are members, test rows aren't.
    mia_agg = aggregated_mia_auc(
        proba_member=proba_train,
        y_member=prepared.y_train,
        proba_non=proba_test,
        y_non=prepared.y_test,
        seed=seed,
    )
    mia_gap = gap_mia_auc(
        proba_member=proba_train,
        y_member=prepared.y_train,
        proba_non=proba_test,
        y_non=prepared.y_test,
    )

    return AEPipelineResult(
        ae=ae_name,
        epsilon=epsilon,
        mechanism=mechanism,
        accuracy=float(accuracy_score(prepared.y_test, y_pred)),
        mia_aggregated_auc=mia_agg.auc,
        mia_gap_auc=mia_gap.auc,
        realised_epsilon=realised_eps,
        n_train=len(prepared.y_train),
        n_test=len(prepared.y_test),
    )
