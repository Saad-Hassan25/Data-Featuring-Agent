"""Static leakage validation + empirical suspicion heuristics (§6.3)."""

from __future__ import annotations

import pytest

from feature_agent.config import FeatureAgentConfig
from feature_agent.guards import (
    InvalidCandidate,
    LeakageError,
    is_empirically_suspicious,
    validate_spec_static,
)
from feature_agent.profile import build_profile
from feature_agent.sample_data import add_leak_columns, make_churn_sample
from feature_agent.schemas import FeatureSpec


def _profile(cfg):
    df = add_leak_columns(make_churn_sample(n=800, seed=1), "churned")
    return build_profile(df, "churned", "classification", cfg)


def test_reference_to_forbidden_column_is_leaky():
    cfg = FeatureAgentConfig(model="x", forbidden_columns=["account_closed"], use_llm=False)
    prof = _profile(cfg)
    spec = FeatureSpec(name="leak_feat", kind="op", op="freq_encode", inputs=["account_closed"])
    with pytest.raises(LeakageError):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_reference_to_target_is_leaky():
    cfg = FeatureAgentConfig(model="x", use_llm=False)
    prof = _profile(cfg)
    spec = FeatureSpec(name="uses_target", kind="op", op="log1p", inputs=["churned"])
    with pytest.raises(LeakageError):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_group_stat_on_near_unique_key_is_leaky():
    cfg = FeatureAgentConfig(model="x", use_llm=False)  # customer_id is near-unique
    prof = _profile(cfg)
    spec = FeatureSpec(name="gs_leak", kind="op", op="group_stat",
                       inputs=["customer_id", "monthly_spend"], params={"stat": "mean"})
    with pytest.raises(LeakageError):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_unknown_column_rejected():
    cfg = FeatureAgentConfig(model="x", use_llm=False)
    prof = _profile(cfg)
    spec = FeatureSpec(name="nope", kind="op", op="log1p", inputs=["does_not_exist"])
    with pytest.raises(InvalidCandidate):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_name_collision_rejected():
    cfg = FeatureAgentConfig(model="x", use_llm=False)
    prof = _profile(cfg)
    spec = FeatureSpec(name="monthly_spend", kind="op", op="log1p", inputs=["days_since_login"])
    with pytest.raises(InvalidCandidate):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_target_encode_disabled_by_default():
    cfg = FeatureAgentConfig(model="x", use_llm=False, enable_target_encode=False)
    prof = _profile(cfg)
    spec = FeatureSpec(name="te_region", kind="op", op="target_encode", inputs=["region"])
    with pytest.raises(InvalidCandidate):
        validate_spec_static(spec, prof, cfg, existing_names=set())


def test_empirical_flag_on_near_target_feature():
    cfg = FeatureAgentConfig(model="x", use_llm=False)
    flagged, why = is_empirically_suspicious(0.99, 0.5, 0.01, cfg, "classification")
    assert flagged and "leakage" in why.lower()


def test_no_flag_on_strong_but_legit_feature():
    cfg = FeatureAgentConfig(model="x", use_llm=False)
    # big marginal lift but only moderate single-feature correlation -> NOT flagged
    flagged, _ = is_empirically_suspicious(0.66, 5.0, 0.05, cfg, "regression")
    assert not flagged
