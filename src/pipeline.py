"""
End-to-end runner:  python -m src.pipeline run  (or `prepare`)

    prepare : raw Kaggle CSVs  ->  data/processed/*.parquet (with features)
    run     : processed data   ->  scores -> ensemble -> submission + figures

The notebooks/ folder holds the full experiment trail (isolation forest, ECOD,
autoencoder, temporal ECOD, CTST, several ensembles). This script reproduces the
practical core of the winning recipe: a rank-average of the strongest scorers.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import ENGINEERED, build_features
from src.models import combine, rank_average, score_ecod, score_isolation_forest
from src.viz import plot_feature_importance, plot_score_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lob")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "outputs" / "submissions"
FIGURES = ROOT / "reports" / "figures"


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


def prepare() -> None:
    """Build feature-augmented parquets from the raw Kaggle CSVs."""
    cfg = _cfg()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_cols = cfg["features"]["raw"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    for split, fname in cfg["data"].items():
        df = pd.read_csv(raw_dir / fname)
        df[raw_cols] = df[raw_cols].fillna(0.0)
        out = PROCESSED / f"{split.replace('_file', '')}.parquet"
        build_features(df).to_parquet(out, index=False)
        log.info("wrote %s (%d rows)", out.name, len(df))


def _matrix(df: pd.DataFrame) -> np.ndarray:
    """Engineered-feature matrix. Infinities go to the edges, not to zero: the
    numpy default of 0.0 lands them in the middle of a standardised feature."""
    X = df[ENGINEERED].to_numpy(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def run(sample: int | None = None, with_autoencoder: bool = False) -> None:
    """Score the test set and write a submission + diagnostic figures.

    Cross-sectional features only. `build_temporal_features` and the rolling
    windows in features.TEMPORAL are used by the notebooks, not by this script -
    they need the rows in per-symbol time order and this path subsamples.
    """
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    if sample:
        train = train.sample(min(sample, len(train)), random_state=42)
        test = test.sample(min(sample, len(test)), random_state=42)
    log.info("train %s | test %s", train.shape, test.shape)

    Xtr, Xte = _matrix(train), _matrix(test)

    cfg = _cfg()
    scores = {
        "isolation_forest": score_isolation_forest(Xtr, Xte, cfg["models"]["isolation_forest"]),
        "ecod": score_ecod(Xtr, Xte, cfg["models"]["ecod"]),
    }
    if with_autoencoder:
        from src.models import score_autoencoder  # lazy: needs torch

        # The validation split has to be held out. `train.sample(frac=0.2)` drew
        # rows that stayed in Xtr, so every one of them was in the training set and
        # early stopping was watching training loss - it would keep improving and
        # the patience counter would never fire on real overfitting. Prefer the
        # competition's own val split; otherwise carve one out and train on the rest.
        val_path = PROCESSED / "val.parquet"
        if val_path.exists():
            X_fit, X_val = Xtr, _matrix(pd.read_parquet(val_path))
        else:
            held = train.sample(frac=0.2, random_state=0).index
            X_fit, X_val = _matrix(train.drop(index=held)), _matrix(train.loc[held])
            log.info("no val.parquet - held out %d rows from train for early stopping", len(held))
        scores["autoencoder"] = score_autoencoder(X_fit, X_val, Xte, cfg["models"]["autoencoder"])

    log.info("scorers: %s", ", ".join(scores))
    # Submission ranks by rank-average (only ordering is scored). For diagnostics we
    # keep the weighted ensemble score, which preserves magnitude and the anomaly tail.
    final = rank_average(*scores.values())
    weighted = combine(scores, cfg["ensemble"]["weights"])

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    # The competition keys rows by a 0-based id; in the processed test set that's
    # the 'index' column ('ID' in train). Fall back to positional if neither exists.
    id_col = next((c for c in ("ID", "index") if c in test.columns), None)
    ids = test[id_col].to_numpy() if id_col else np.arange(len(test))
    sub = pd.DataFrame({"ID": ids, "TARGET": final})
    sub_path = SUBMISSIONS / "submission_pipeline.csv"
    sub.to_csv(sub_path, index=False)
    log.info("submission -> %s (%d rows, %d flagged @0.95)",
             sub_path.name, len(sub), int((final > np.quantile(final, 0.95)).sum()))

    plot_score_distribution(weighted, FIGURES / "score_distribution.png")
    plot_feature_importance(test, weighted, ENGINEERED, FIGURES / "feature_importance.png")
    log.info("figures -> %s", FIGURES)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LOB market-manipulation anomaly pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="build processed parquets from raw CSVs")
    r = sub.add_parser("run", help="score, ensemble, write submission + figures")
    r.add_argument("--sample", type=int, default=None, help="subsample N rows for a quick run")
    r.add_argument("--with-autoencoder", action="store_true", help="include the torch autoencoder scorer")

    args = p.parse_args(argv)
    if args.cmd == "prepare":
        prepare()
    else:
        run(sample=args.sample, with_autoencoder=args.with_autoencoder)


if __name__ == "__main__":
    main()
