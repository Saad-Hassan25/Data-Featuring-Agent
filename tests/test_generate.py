"""Generation: deterministic candidates are valid, unique, and dedupe across rounds."""

from __future__ import annotations

from feature_agent.config import FeatureAgentConfig
from feature_agent.executor import dry_run
from feature_agent.generate import dedup, deterministic_candidates
from feature_agent.guards import InvalidCandidate, allowed_columns, validate_spec_static
from feature_agent.profile import build_profile
from feature_agent.sample_data import make_churn_sample


def _setup():
    cfg = FeatureAgentConfig(model="x", use_llm=False, forbidden_columns=["cancellation_date"],
                             group_column="customer_id", n_candidates_per_round=20)
    df = make_churn_sample(n=800, seed=3)
    prof = build_profile(df, "churned", "classification", cfg)
    return df, cfg, prof


def test_deterministic_candidates_reference_only_allowed_columns():
    df, cfg, prof = _setup()
    allowed = allowed_columns(prof, cfg)
    specs = deterministic_candidates(prof, cfg)
    assert len(specs) >= 10
    for s in specs:
        for col in s.inputs:
            assert col in allowed, f"{s.name} references disallowed column {col}"


def test_candidate_validity_rate_at_least_80pct():
    """Success metric: >=80% of generated specs validate and execute (§11)."""
    df, cfg, prof = _setup()
    allowed = allowed_columns(prof, cfg)
    specs = deterministic_candidates(prof, cfg)
    ok = 0
    for s in specs:
        try:
            validate_spec_static(s, prof, cfg, existing_names=set())
            dry_run(s, df, allowed, y=df["churned"])
            ok += 1
        except InvalidCandidate:
            pass
    assert ok / len(specs) >= 0.80


def test_dedup_removes_seen_and_duplicates():
    df, cfg, prof = _setup()
    specs = deterministic_candidates(prof, cfg)
    seen = {specs[0].signature()}
    out = dedup(specs, seen)
    sigs = [s.signature() for s in out]
    assert specs[0].signature() not in sigs           # seen removed
    assert len(sigs) == len(set(sigs))                # no duplicates within batch


def test_signature_is_name_independent():
    df, cfg, prof = _setup()
    specs = deterministic_candidates(prof, cfg)
    s = specs[0]
    twin = s.model_copy(update={"name": "renamed_twin"})
    assert s.signature() == twin.signature()          # same computation, same signature
