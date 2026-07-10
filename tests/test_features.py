import numpy as np
import pandas as pd
import pytest

from src.features import (
    ENGINEERED,
    TEMPORAL,
    build_features,
    build_temporal_features,
    _safe_div,
)

RAW_COLS = [
    "ReturnBid1", "ReturnAsk1", "DerivativeReturnBid1", "DerivativeReturnAsk1",
    "BidSize1", "AskSize1", "TradeBidSize", "TradeAskSize",
    "CancelledBidSize", "CancelledAskSize",
    "TradeBidIndicator", "TradeAskIndicator",
    "CancelledBidIndicator", "CancelledAskIndicator",
]


@pytest.fixture
def lob(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: rng.normal(size=n) for k in RAW_COLS})
    df[["BidSize1", "AskSize1", "TradeBidSize", "TradeAskSize",
        "CancelledBidSize", "CancelledAskSize"]] = np.abs(
        df[["BidSize1", "AskSize1", "TradeBidSize", "TradeAskSize",
            "CancelledBidSize", "CancelledAskSize"]]
    )
    df["ExternalSymbol"] = np.where(np.arange(n) % 2 == 0, "AAA", "BBB")
    df["TimeInMilliSecs"] = np.arange(n) * 100
    df["OriginalSequenceNumber"] = np.arange(n)
    return df


def test_all_engineered_columns_present(lob):
    out = build_features(lob)
    missing = [c for c in ENGINEERED if c not in out.columns]
    assert not missing
    assert len(ENGINEERED) == 27  # the number the README claims


def test_no_infs_or_nans(lob):
    # zero denominators must not produce inf (thin-book rows)
    lob.loc[0, ["TradeBidSize", "TradeAskSize", "CancelledBidSize", "CancelledAskSize"]] = 0.0
    out = build_features(lob)
    eng = out[ENGINEERED]
    assert np.isfinite(eng.to_numpy()).all()


def test_safe_div_zero_denominator():
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 4.0])
    out = _safe_div(a, b)
    assert out[0] == 0.0
    assert out[1] == 0.5


def test_cancel_trade_ratio_flags_spoofy_row(lob):
    # heavy cancels + no trades = classic spoofing signature
    lob.loc[5, ["CancelledBidSize", "CancelledAskSize"]] = 50.0
    lob.loc[5, ["TradeBidSize", "TradeAskSize"]] = 0.01
    out = build_features(lob)
    assert out.loc[5, "cancel_trade_ratio"] > out["cancel_trade_ratio"].median()


def test_temporal_columns_and_causality(lob):
    out = build_temporal_features(build_features(lob))
    assert all(c in out.columns for c in TEMPORAL)
    # rolling windows are per symbol and causal: first row of each symbol
    # can only see itself
    first_rows = out.groupby("ExternalSymbol").head(1)
    total = first_rows["CancelledBidSize"] + first_rows["CancelledAskSize"]
    assert np.allclose(first_rows["roll_cancel_5"].to_numpy(), total.to_numpy())


def test_time_delta_clipped(lob):
    lob.loc[100, "TimeInMilliSecs"] = 10_000_000  # day-boundary style jump
    out = build_temporal_features(build_features(lob))
    assert out["time_delta"].max() <= 60_000
