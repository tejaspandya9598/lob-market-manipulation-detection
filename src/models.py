"""
One-class anomaly scorers and the ensemble that combines them.

The competition is unsupervised: there are no manipulation labels at train time,
so every model here learns "what normal order-book activity looks like" and scores
each test row by how far it deviates. Four scorers, deliberately different in what
they treat as anomalous:

  * Isolation Forest — random axis-aligned splits; anomalies isolate in fewer cuts.
  * ECOD            — tail probability under each feature's empirical CDF; fast,
                       parameter-light, and the strongest single model here.
  * ECOD per symbol — same, but fit separately per instrument so a quiet name and
                       a busy name aren't judged on the same scale.
  * Autoencoder     — reconstruction error from a network trained on normal rows.

Every scorer returns values normalised to roughly [0, 1] against the *training*
distribution (see `ref_norm`), which is what makes them safe to average.
"""
from __future__ import annotations

import numpy as np
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def ref_norm(scores: np.ndarray, train_mean: float, train_std: float) -> np.ndarray:
    """Squash raw scores through a sigmoid centred on the *training* mean/std.

    Using train statistics (not test) keeps the mapping honest: a test point only
    scores high if it's unusual relative to what the model saw as normal.
    """
    return 1.0 / (1.0 + np.exp(-((scores - train_mean) / (train_std + 1e-10))))


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo < 1e-10 else (scores - lo) / (hi - lo)


# --------------------------------------------------------------------------- IF
def score_isolation_forest(X_train, X_test, params, seed: int = 42) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    model = IsolationForest(random_state=seed, **params).fit(Xtr)
    # decision_function is high-for-normal, so negate to get high-for-anomaly.
    train_raw = -model.decision_function(Xtr)
    return ref_norm(-model.decision_function(Xte), float(train_raw.mean()), float(train_raw.std()))


# ------------------------------------------------------------------------- ECOD
def score_ecod(X_train, X_test, params) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    model = ECOD(**params).fit(Xtr)
    train_raw = model.decision_function(Xtr)
    return ref_norm(model.decision_function(Xte), float(train_raw.mean()), float(train_raw.std()))


def score_ecod_per_symbol(train_df, test_df, params, feature_cols, symbol_col="ExternalSymbol") -> np.ndarray:
    """Fit a separate ECOD per instrument. A symbol needs >=100 training rows to
    get its own model; otherwise its test rows keep a score of 0 (treated normal)."""
    scores = np.zeros(len(test_df), dtype=np.float64)
    tr_sym, te_sym = train_df[symbol_col].to_numpy(), test_df[symbol_col].to_numpy()

    for sym in np.unique(tr_sym):
        tr_mask, te_mask = tr_sym == sym, te_sym == sym
        if te_mask.sum() == 0 or tr_mask.sum() < 100:
            continue

        X_tr = train_df.loc[tr_mask, feature_cols].to_numpy(np.float32, na_value=0.0)
        X_te = test_df.loc[te_mask, feature_cols].to_numpy(np.float32, na_value=0.0)
        for arr in (X_tr, X_te):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler().fit(X_tr)
        model = ECOD(**params).fit(scaler.transform(X_tr))
        train_raw = model.decision_function(scaler.transform(X_tr))
        scores[te_mask] = ref_norm(model.decision_function(scaler.transform(X_te)),
                                   float(train_raw.mean()), float(train_raw.std()))
    return scores


# ------------------------------------------------------------------ Autoencoder
def _resolve_device():
    """Best torch device available: CUDA, then Apple MPS, then CPU. Imported
    lazily so the IF/ECOD path doesn't require torch to be installed."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_autoencoder(n_features: int, hidden: list[int], dropout: float):
    """Symmetric dense autoencoder. Defined inside a function so `import torch`
    stays out of module scope."""
    import torch.nn as nn

    layers, prev = [], n_features
    for h in hidden:  # encoder narrows then decoder widens — symmetric hidden_dims
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_features))  # reconstruct back to input width
    return nn.Sequential(*layers)


def _reconstruction_errors(net, scaler, X, device, batch_size: int = 512) -> np.ndarray:
    import torch

    net.eval()
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
    errors = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            b = Xs[i:i + batch_size].to(device)
            errors.append(((net(b) - b) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errors)


def score_autoencoder(X_train, X_val, X_test, p) -> np.ndarray:
    """Train an autoencoder on normal rows, score test rows by reconstruction
    error. Early-stops on the validation loss with `p['patience']`."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = _resolve_device()
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32).to(device)

    net = _build_autoencoder(Xt.shape[1], p["hidden_dims"], p["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    loader = DataLoader(TensorDataset(Xt, Xt), batch_size=p["batch_size"], shuffle=True)

    best_loss, since_improved = float("inf"), 0
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    for _ in range(p["epochs"]):
        net.train()
        for (xb,) in loader:
            xb = xb.to(device)
            loss = ((net(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = ((net(Xv) - Xv) ** 2).mean().item()
        if val_loss < best_loss:
            best_loss, since_improved = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        elif (since_improved := since_improved + 1) >= p["patience"]:
            break

    net.load_state_dict(best_state)
    train_raw = _reconstruction_errors(net, scaler, X_train, device)
    return ref_norm(_reconstruction_errors(net, scaler, X_test, device),
                    float(train_raw.mean()), float(train_raw.std()))


# --------------------------------------------------------------------- ensemble
def combine(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of per-model scores, renormalised to [0, 1]."""
    keys = list(scores)
    matrix = np.stack([scores[k] for k in keys], axis=0)
    w = np.array([weights.get(k, 0.0) for k in keys], dtype=np.float64)
    w /= w.sum()
    return minmax(w @ matrix)


def rank_average(*score_arrays: np.ndarray) -> np.ndarray:
    """Average the *ranks* of several score vectors (what the final submission
    used). Robust to scale differences between models — only ordering counts."""
    from scipy.stats import rankdata

    n = len(score_arrays[0])
    ranks = np.sum([rankdata(s) for s in score_arrays], axis=0)
    return ranks / (len(score_arrays) * n)
