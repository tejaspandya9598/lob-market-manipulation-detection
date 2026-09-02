import numpy as np
import pytest

from src.models import combine, minmax, rank_average, ref_norm, score_isolation_forest


def test_ref_norm_uses_train_stats():
    train = np.array([0.0, 1.0, 2.0])
    # scores exactly at the train mean map to 0.5; far above map toward 1
    out = ref_norm(np.array([1.0, 100.0]), train.mean(), train.std())
    assert abs(out[0] - 0.5) < 1e-9
    assert out[1] > 0.99


def test_minmax_bounds():
    out = minmax(np.array([3.0, 7.0, 11.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_combine_weights_and_renormalises():
    s = {"a": np.array([0.0, 0.5, 1.0]), "b": np.array([1.0, 0.5, 0.0])}
    out = combine(s, {"a": 0.75, "b": 0.25})
    # output is minmax-rescaled to [0, 1] and ordered by the dominant model
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.all(np.diff(out) > 0)  # 'a' has 3x the weight, so its order wins


def test_rank_average_is_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    b = a * 1000.0  # same ordering, wildly different scale
    out1 = rank_average(a, b)
    out2 = rank_average(a, a)
    assert np.allclose(out1, out2)


def test_isolation_forest_scores_obvious_outlier():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(500, 5))
    X_test = np.vstack([rng.normal(size=(50, 5)), np.full((1, 5), 10.0)])
    scores = score_isolation_forest(
        X_train, X_test, {"n_estimators": 100, "contamination": 0.01}
    )
    assert scores[-1] == pytest.approx(scores.max())


def test_combine_refuses_a_weightless_ensemble():
    """`w /= w.sum()` on an all-zero weight vector wrote a submission of NaN
    behind nothing louder than a RuntimeWarning."""
    with pytest.raises(ValueError, match="no ensemble weight matches"):
        combine({"ecod": np.array([1.0, 2.0])}, {"something_else": 1.0})


def test_combine_renormalises_over_the_scorers_that_ran():
    out = combine({"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])},
                  {"a": 0.2, "b": 0.6, "c": 0.2})   # c never ran
    assert out.min() == 0.0 and out.max() == 1.0


def test_unseen_symbols_are_scored_not_waved_through():
    """A symbol absent from training kept a score of 0.0 - the most normal value
    the scorer can return - and a brand-new instrument is exactly where you would
    look for manipulation."""
    import pandas as pd

    from src.models import score_ecod_per_symbol

    rng = np.random.default_rng(0)
    cols = ["f1", "f2"]
    train = pd.DataFrame({"ExternalSymbol": ["A"] * 200,
                          "f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    test = pd.DataFrame({
        "ExternalSymbol": ["A"] * 5 + ["NEW"] * 5,
        "f1": list(rng.normal(size=5)) + [50.0, 60.0, 70.0, 80.0, 90.0],
        "f2": list(rng.normal(size=5)) + [50.0, 60.0, 70.0, 80.0, 90.0],
    })
    scores = score_ecod_per_symbol(train, test, {"contamination": 0.05}, cols)
    assert np.isfinite(scores).all()
    assert (scores[5:] > 0.9).all()          # wild outliers score high
    assert scores[5:].min() > scores[:5].max()   # and above the in-sample rows


def test_infinities_do_not_land_in_the_middle():
    """np.nan_to_num sends +/-inf to 0.0, which after standardising is the centre
    of the distribution."""
    import pandas as pd

    from src.models import _clean

    df = pd.DataFrame({"f": [np.inf, -np.inf, np.nan, 1.0]})
    out = _clean(df, np.ones(4, dtype=bool), ["f"])[:, 0]
    assert out[0] > 1e5 and out[1] < -1e5 and out[2] == 0.0
