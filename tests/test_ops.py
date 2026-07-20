"""Op library: correct materialization and train-only fitting (fold safety)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_agent.guards import InvalidCandidate
from feature_agent.ops import build_op


def _df():
    return pd.DataFrame({
        "num": [1.0, 2.0, 4.0, 8.0, np.nan],
        "den": [1.0, 0.0, 2.0, 4.0, 2.0],
        "cat": ["a", "a", "b", "c", "b"],
        "date": pd.to_datetime(["2021-01-02", "2021-06-15", "2021-12-25", "2022-03-01", "2021-07-04"]),
    })


def test_ratio_handles_zero_denominator():
    out = build_op("ratio", "r", ["num", "den"], {}).fit(_df()).transform(_df())
    assert np.isfinite(out.to_numpy()[1])  # 2/(0+eps) is finite, not inf


def test_log1p_negative_is_nan_not_crash():
    df = pd.DataFrame({"x": [-2.0, 0.0, 3.0]})
    out = build_op("log1p", "l", ["x"], {}).fit(df).transform(df)
    assert np.isnan(out.to_numpy()[0]) and abs(out.to_numpy()[2] - np.log1p(3.0)) < 1e-9


def test_date_part_is_weekend():
    out = build_op("date_part", "w", ["date"], {"part": "is_weekend"}).fit(_df()).transform(_df())
    # 2021-06-15 is a Tuesday (0), 2021-12-25 is a Saturday (1)
    assert out.to_numpy()[1] == 0.0 and out.to_numpy()[2] == 1.0


def test_count_encode_is_fit_on_training_only():
    train = _df().iloc[:3]      # cat: a, a, b
    val = _df().iloc[3:]        # cat: c, b
    op = build_op("count_encode", "ce", ["cat"], {}).fit(train)
    out = op.transform(val)
    # 'c' unseen in train -> 0; 'b' seen once in train -> 1
    assert list(out.to_numpy()) == [0.0, 1.0]


def test_group_stat_uses_train_groups():
    train = pd.DataFrame({"cat": ["a", "a", "b", "b"], "num": [10.0, 20.0, 100.0, 200.0]})
    op = build_op("group_stat", "gs", ["cat", "num"], {"stat": "mean"}).fit(train)
    out = op.transform(pd.DataFrame({"cat": ["a", "b", "z"], "num": [0, 0, 0]}))
    assert out.to_numpy()[0] == 15.0 and out.to_numpy()[1] == 150.0  # z -> global fallback


def test_bin_quantile_edges_from_train():
    train = pd.DataFrame({"x": np.arange(100.0)})
    op = build_op("bin_quantile", "bq", ["x"], {"n_bins": 4}).fit(train)
    out = op.transform(pd.DataFrame({"x": [0.0, 50.0, 99.0]}))
    assert out.to_numpy()[0] < out.to_numpy()[1] < out.to_numpy()[2]


def test_arity_validation():
    with pytest.raises(InvalidCandidate):
        build_op("ratio", "r", ["num"], {})           # ratio needs 2 inputs
    with pytest.raises(InvalidCandidate):
        build_op("date_part", "d", ["date"], {"part": "nonsense"})
    with pytest.raises(InvalidCandidate):
        build_op("unknown_op", "u", ["num"], {})
