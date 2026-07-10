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
