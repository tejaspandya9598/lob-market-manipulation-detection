"""
Feature engineering for limit-order-book rows.

The raw Kaggle columns are per-event LOB primitives (bid/ask returns, sizes, trade
and cancel flows). Manipulation — spoofing, layering, momentum ignition — shows up
not in any single column but in *imbalances* and *interactions*: lots of cancels
relative to trades, size stacked on one side, returns that move with order flow.

Two feature sets:
  * `build_features`           — 27 cross-sectional features computed row-by-row.
  * `build_temporal_features`  — rolling windows per instrument, for the models
                                  that exploit sequence (the strongest ones did).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Cross-sectional engineered features (order matters for reproducibility).
ENGINEERED = [
    "spread", "spread_abs", "mid_return", "return_imbalance",
    "deriv_spread", "deriv_mid", "deriv_imbalance",
    "size_imbalance", "size_imbalance_ratio",
    "trade_imbalance", "trade_imbalance_ratio",
    "cancel_imbalance", "cancel_imbalance_ratio",
    "total_cancel", "total_trade", "cancel_trade_ratio",
    "trade_indicator_imbalance", "cancel_indicator_imbalance", "indicator_cross",
    "return_x_size", "spread_x_cancel", "deriv_x_trade",
    "return_bid_sq", "return_ask_sq", "size_imbalance_sq",
    "abs_return_bid", "abs_return_ask",
]


def _safe_div(a, b):
    """Divide, returning 0 where the denominator is ~0 (avoids inf on thin books)."""
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-8)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 27 cross-sectional features. Pulls columns into float64 arrays
    once and works vectorised — at ~1M rows the per-column overhead matters."""
    c = {k: df[k].to_numpy(dtype=np.float64) for k in (
        "ReturnBid1", "ReturnAsk1", "DerivativeReturnBid1", "DerivativeReturnAsk1",
        "BidSize1", "AskSize1", "TradeBidSize", "TradeAskSize",
        "CancelledBidSize", "CancelledAskSize",
        "TradeBidIndicator", "TradeAskIndicator",
        "CancelledBidIndicator", "CancelledAskIndicator",
    )}
    rb, ra = c["ReturnBid1"], c["ReturnAsk1"]
    drb, dra = c["DerivativeReturnBid1"], c["DerivativeReturnAsk1"]
    bs, as_ = c["BidSize1"], c["AskSize1"]
    tb, ta = c["TradeBidSize"], c["TradeAskSize"]
    cb, ca = c["CancelledBidSize"], c["CancelledAskSize"]
    tib, tia = c["TradeBidIndicator"], c["TradeAskIndicator"]
    cib, cia = c["CancelledBidIndicator"], c["CancelledAskIndicator"]

    spread = ra - rb               # bid/ask return gap
    mid = (rb + ra) * 0.5          # mid-price return proxy
    dmid = (drb + dra) * 0.5       # its first derivative (acceleration)
    si = bs - as_                  # depth imbalance (book pressure)
    ti = tb - ta                   # trade-flow imbalance
    ci = cb - ca                   # cancel-flow imbalance (the spoofing tell)
    tc, tt = cb + ca, tb + ta      # total cancels / total trades

    eng = {
        "spread": spread, "spread_abs": np.abs(spread),
        "mid_return": mid, "return_imbalance": rb - ra,
        "deriv_spread": dra - drb, "deriv_mid": dmid, "deriv_imbalance": drb - dra,
        "size_imbalance": si,
        "size_imbalance_ratio": _safe_div(si, np.abs(bs) + np.abs(as_)),
        "trade_imbalance": ti,
        "trade_imbalance_ratio": _safe_div(ti, np.abs(tb) + np.abs(ta)),
        "cancel_imbalance": ci,
        "cancel_imbalance_ratio": _safe_div(ci, np.abs(cb) + np.abs(ca)),
        "total_cancel": tc, "total_trade": tt,
        # cancel-to-trade ratio: high values are the classic spoofing signature
        "cancel_trade_ratio": _safe_div(tc, np.abs(tt)),
        "trade_indicator_imbalance": tib - tia,
        "cancel_indicator_imbalance": cib - cia,
        "indicator_cross": tib * cib,
        # interactions: a move that coincides with one-sided flow is more suspect
        "return_x_size": mid * si,
        "spread_x_cancel": spread * tc,
        "deriv_x_trade": dmid * ti,
        # non-linear terms give the tree/Isolation models curvature to split on
        "return_bid_sq": rb * rb, "return_ask_sq": ra * ra,
        "size_imbalance_sq": si * si,
        "abs_return_bid": np.abs(rb), "abs_return_ask": np.abs(ra),
    }
    return pd.concat([df, pd.DataFrame(eng, index=df.index)], axis=1)


def feature_names(raw: list[str]) -> list[str]:
    return list(raw) + ENGINEERED


# Per-instrument rolling features (sequence context).
TEMPORAL = [
    "time_delta",
    "roll_cancel_5", "roll_cancel_20", "roll_cancel_100",
    "roll_trade_5", "roll_trade_20", "roll_trade_100",
    "roll_retstd_5", "roll_retstd_20", "roll_retstd_100",
    "roll_absret_max_20", "roll_absret_max_100",
    "roll_cancel_trade_ratio_20",
]


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling cancel/trade/volatility windows per instrument. Spoofing is
    bursty, so a window that's suddenly cancel-heavy is a stronger signal than any
    single row. Sorted by (symbol, time, sequence) so the windows are causal."""
    df = df.sort_values(["ExternalSymbol", "TimeInMilliSecs", "OriginalSequenceNumber"]).reset_index(drop=True)
    df["_cancel"] = df["CancelledBidSize"].to_numpy() + df["CancelledAskSize"].to_numpy()
    df["_trade"] = df["TradeBidSize"].to_numpy() + df["TradeAskSize"].to_numpy()
    df["_absret"] = np.abs(df["ReturnBid1"].to_numpy())

    g = df.groupby("ExternalSymbol", sort=False)
    # inter-event gap, clipped to [0, 60s] so day-boundary jumps don't blow up
    df["time_delta"] = g["TimeInMilliSecs"].diff().fillna(0).clip(lower=0, upper=60000)

    def _roll(col, w, fn):
        s = getattr(g[col].rolling(w, min_periods=1), fn)()
        return s.reset_index(level=0, drop=True)

    for w in (5, 20, 100):
        df[f"roll_cancel_{w}"] = _roll("_cancel", w, "sum")
        df[f"roll_trade_{w}"] = _roll("_trade", w, "sum")
        df[f"roll_retstd_{w}"] = _roll("ReturnBid1", w, "std").fillna(0)

    for w in (20, 100):
        df[f"roll_absret_max_{w}"] = _roll("_absret", w, "max")

    df["roll_cancel_trade_ratio_20"] = df["roll_cancel_20"] / (df["roll_cancel_20"] + df["roll_trade_20"] + 1e-8)

    return df.drop(columns=["_cancel", "_trade", "_absret"])


def feature_names_all(raw: list[str]) -> list[str]:
    return list(raw) + ENGINEERED + TEMPORAL
