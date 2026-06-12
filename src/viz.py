"""Diagnostic plots for the anomaly scores."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_score_distribution(scores: np.ndarray, out_path: str | Path, flag_quantile: float = 0.95) -> None:
    """Histogram of ensemble scores with the flag threshold marked."""
    thr = float(np.quantile(scores, flag_quantile))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=80, color="#3b6ea5", alpha=0.85)
    ax.axvline(thr, color="#c0392b", ls="--", lw=1.5, label=f"{flag_quantile:.0%} pct = {thr:.2f} (flagged)")
    ax.set_title("Ensemble anomaly-score distribution", fontweight="bold")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("rows")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_feature_importance(test_df: pd.DataFrame, scores: np.ndarray, feature_cols: list[str],
                            out_path: str | Path, top_n: int = 12) -> None:
    """Rank features by absolute correlation with the anomaly score — a cheap,
    model-agnostic read on what's driving the flags."""
    corr = {f: abs(np.corrcoef(np.nan_to_num(test_df[f].to_numpy(float)), scores)[0, 1]) for f in feature_cols}
    top = sorted(corr, key=corr.get, reverse=True)[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top, [corr[f] for f in top], color="#2c7a4b")
    ax.set_title("Top features by |correlation| with anomaly score", fontweight="bold")
    ax.set_xlabel("|correlation|")
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
