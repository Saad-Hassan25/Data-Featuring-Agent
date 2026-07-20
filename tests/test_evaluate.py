"""Evaluation harness recovery benchmark (M3 / §10).

On a synthetic dataset with a planted, base-invisible signal (a date-derived
feature) and several noise candidates, the planted feature must beat the shadow
gate across seeds while the noise candidates fail it (false-keep rate small)."""

from __future__ import annotations

from feature_agent.config import FeatureAgentConfig
from feature_agent.evaluate import Evaluator
from feature_agent.profile import build_profile
from feature_agent.sample_data import make_planted_regression
from feature_agent.schemas import FeatureSpec

PLANTED = FeatureSpec(name="dow_signal", kind="op", op="date_part",
                      inputs=["event_date"], params={"part": "is_weekend"})
NOISE = [
    FeatureSpec(name="n_z1z2", kind="op", op="ratio", inputs=["z1", "z2"]),
    FeatureSpec(name="n_z3z4", kind="op", op="ratio", inputs=["z3", "z4"]),
    FeatureSpec(name="n_z2z5", kind="op", op="product", inputs=["z2", "z5"]),
    FeatureSpec(name="n_z1z5", kind="op", op="product", inputs=["z1", "z5"]),
    FeatureSpec(name="n_z4z3", kind="op", op="difference", inputs=["z4", "z3"]),
]


def _passes(ev, name: str) -> bool:
    return ev.importances[name] > ev.shadow_ceiling and ev.shadow_percentile[name] >= 95.0


def _run_seed(seed: int):
    cfg = FeatureAgentConfig(model="x", use_llm=False, n_folds=5, random_state=seed)
    df = make_planted_regression(n=1200, seed=seed)
    prof = build_profile(df, "target", "regression", cfg)
    ev = Evaluator(df, prof, cfg)
    result = ev.evaluate_candidates([PLANTED] + NOISE)
    return ev.orient, result


def test_recovery_and_false_keep_rate():
    seeds = [0, 1, 2, 3, 4]
    recovered = 0
    false_keeps = 0
    for s in seeds:
        orient, res = _run_seed(s)
        if _passes(res, PLANTED.name):
            recovered += 1
        false_keeps += sum(_passes(res, n.name) for n in NOISE)
        # the planted feature should reduce error (augmented better than baseline)
        assert orient * (res.augmented_mean - res.baseline_mean) > 0
    assert recovered >= 4, f"planted feature recovered in only {recovered}/5 seeds"
    total_noise = len(NOISE) * len(seeds)
    assert false_keeps / total_noise <= 0.10, f"false-keep rate {false_keeps}/{total_noise} too high"


def test_shadow_ceiling_positive_and_planted_dominates():
    _, res = _run_seed(0)
    assert res.importances[PLANTED.name] > max(res.importances[n.name] for n in NOISE)
