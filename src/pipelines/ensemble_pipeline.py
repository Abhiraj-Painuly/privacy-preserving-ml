"""Ensemble pipeline (paper Fig 1b, §III-B).

Trains four MLP backbones (ResNet, DenseNet, WideResNet, PreResNet) plus
DP-XGBoost on the perturbed feature matrix, and fuses their predictions
either via Confident GNMax or confidence-based fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..attacks.mia_aggregated import aggregated_mia_auc
from ..attacks.mia_gap import gap_mia_auc
from ..models.aggregation import confidence_fusion, confident_gnmax
from ..models.backbones import (
    DenseNetGN,
    DPXGBoost,
    PreResNetGN,
    ResNetGN,
    WideResNetGN,
)
from ..privacy.dp_sgd import make_private, spent_epsilon
from ..utils import device, set_seed
from ._common import DPMechanismName, prepare

AggMode = Literal["gnmax", "confidence"]


@dataclass
class EnsembleResult:
    epsilon: float
    mechanism: DPMechanismName
    aggregator: AggMode
    accuracy: float
    mia_aggregated_auc: float
    mia_gap_auc: float
    per_backbone_accuracy: dict[str, float]
    realised_epsilon: float


def _build_backbones(input_dim: int, n_classes: int) -> dict[str, nn.Module]:
    return {
        "resnet": ResNetGN(input_dim, n_classes),
        "densenet": DenseNetGN(input_dim, n_classes),
        "wide_resnet": WideResNetGN(input_dim, n_classes),
        "pre_resnet": PreResNetGN(input_dim, n_classes),
    }


def _train_backbone(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    batch_size: int,
    use_dpsgd: bool,
) -> tuple[nn.Module, float]:
    dev = device()
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    realised = target_epsilon
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

    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    if use_dpsgd and state is not None:
        realised = spent_epsilon(state)
    return model, realised


@torch.no_grad()
def _proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    dev = device()
    xb = torch.from_numpy(X.astype(np.float32)).to(dev)
    return torch.softmax(model(xb), dim=-1).cpu().numpy()


def run_ensemble_pipeline(
    df: pd.DataFrame,
    *,
    target: str,
    declared_sensitive: list[str],
    epsilon: float,
    mechanism: DPMechanismName = "laplace",
    delta: float = 1e-5,
    aggregator: AggMode = "confidence",
    use_pca: bool = False,
    epochs: int = 10,
    batch_size: int = 128,
    use_dpsgd: bool = True,
    seed: int = 42,
) -> EnsembleResult:
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
    n_classes = int(np.max(prepared.y_train)) + 1
    input_dim = prepared.X_train.shape[1]

    backbones = _build_backbones(input_dim, n_classes)
    realised = epsilon
    proba_train_list: list[np.ndarray] = []
    proba_test_list: list[np.ndarray] = []
    per_backbone: dict[str, float] = {}

    for name, model in backbones.items():
        trained, eps = _train_backbone(
            model,
            prepared.X_train,
            prepared.y_train,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            batch_size=batch_size,
            use_dpsgd=use_dpsgd,
        )
        realised = max(realised, eps)
        ptr = _proba(trained, prepared.X_train)
        pte = _proba(trained, prepared.X_test)
        proba_train_list.append(ptr)
        proba_test_list.append(pte)
        per_backbone[name] = float(
            accuracy_score(prepared.y_test, pte.argmax(axis=1))
        )

    # DP-XGBoost — operates on numpy directly
    xgb = DPXGBoost(epsilon=epsilon, seed=seed)
    xgb.fit(prepared.X_train, prepared.y_train)
    proba_train_list.append(xgb.predict_proba(prepared.X_train))
    proba_test_list.append(xgb.predict_proba(prepared.X_test))
    per_backbone["dp_xgboost"] = float(
        accuracy_score(prepared.y_test, proba_test_list[-1].argmax(axis=1))
    )

    proba_train = np.stack(proba_train_list, axis=0)  # (T, N, K)
    proba_test = np.stack(proba_test_list, axis=0)

    if aggregator == "confidence":
        fused_test = confidence_fusion(proba_test, mode="weighted", temperature=1.5)
        fused_train = confidence_fusion(proba_train, mode="weighted", temperature=1.5)
        y_pred = fused_test.argmax(axis=1)
    else:
        # GNMax operates on hard labels.
        teacher_train = proba_train.argmax(axis=-1)  # (T, N)
        teacher_test = proba_test.argmax(axis=-1)
        res_test = confident_gnmax(
            teacher_test, n_classes=n_classes, sigma_threshold=5.0, sigma_aggregator=1.0
        )
        # For metric purposes we treat abstained queries as wrong (label != y_test).
        y_pred = np.where(res_test.labels < 0, -1, res_test.labels)
        fused_test = proba_test.mean(axis=0)
        fused_train = proba_train.mean(axis=0)

    accuracy = float(accuracy_score(prepared.y_test, y_pred))
    mia_agg = aggregated_mia_auc(
        proba_member=fused_train,
        y_member=prepared.y_train,
        proba_non=fused_test,
        y_non=prepared.y_test,
        seed=seed,
    )
    mia_gap = gap_mia_auc(
        proba_member=fused_train,
        y_member=prepared.y_train,
        proba_non=fused_test,
        y_non=prepared.y_test,
    )

    return EnsembleResult(
        epsilon=epsilon,
        mechanism=mechanism,
        aggregator=aggregator,
        accuracy=accuracy,
        mia_aggregated_auc=mia_agg.auc,
        mia_gap_auc=mia_gap.auc,
        per_backbone_accuracy=per_backbone,
        realised_epsilon=realised,
    )
