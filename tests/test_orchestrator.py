"""End-to-end orchestration with a stubbed LLM (no network), determinism, and the
full-pipeline leakage canaries (§10)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import feature_agent.orchestrator as orch
from feature_agent.config import FeatureAgentConfig
from feature_agent.sample_data import add_leak_columns, make_churn_sample
from feature_agent.schemas import (
    AdjudicationBatch,
    FeatureReportNarrative,
    GenerationBatch,
)

SENTINEL_SUMMARY = "STUBBED-LLM executive summary."


class _FakeBudget:
    cost_usd = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    def would_exceed(self):
        return False


def make_fake_llm(gen_specs):
    class FakeLLM:
        def __init__(self, config):
            self.config = config
            self.budget = _FakeBudget()
            self.call_log = []

        def available(self):
            return True, ""

        def structured(self, *, stage, system, user, schema, temperature,
                        model=None, max_retries=2):
            if schema is GenerationBatch:
                return GenerationBatch(candidates=list(gen_specs))
            if schema is AdjudicationBatch:
                return AdjudicationBatch(verdicts=[])          # prune borderline by default
            if schema is FeatureReportNarrative:
                return FeatureReportNarrative(
                    executive_summary=SENTINEL_SUMMARY,
                    methodology_note="Marginal lift over baseline via shadow-gated CV.")
            raise AssertionError(f"unexpected schema {schema}")
    return FakeLLM


def _cfg(**kw):
    base = dict(model="stub/model", use_llm=True, max_rounds=2, n_folds=5,
                n_candidates_per_round=10, verbose=False, random_state=0,
                group_column="customer_id", forbidden_columns=["cancellation_date"],
                output_dir=tempfile.mkdtemp(prefix="fa_orch_"))
    base.update(kw)
    return FeatureAgentConfig(**base)


def test_stubbed_llm_end_to_end(monkeypatch):
    gen = [
        {"name": "month_signup", "kind": "op", "op": "date_part",
         "inputs": ["signup_date"], "params": {"part": "month"},
         "rationale": "signup cohort seasonality"},
        {"name": "tickets_per_spend", "kind": "op", "op": "ratio",
         "inputs": ["support_tickets_90d", "monthly_spend"],
         "rationale": "support burden relative to spend"},
        {"name": "spend_expr", "kind": "expression",
         "expression": "monthly_spend / (days_since_login + 1)",
         "rationale": "spend per recency"},
    ]
    monkeypatch.setattr(orch, "LLMClient", make_fake_llm(gen))
    df = make_churn_sample(n=2500, seed=7)
    res = orch.FeatureAgent(_cfg()).run(df, target="churned", task="classification",
                                        domain_context="B2B SaaS churn")
    # artifacts present
    for f in ("manifest.json", "feature_registry.json", "pipeline.joblib", "report.md"):
        assert (Path(res.run_dir) / f).exists(), f"missing {f}"
    assert (Path(res.run_dir) / "rounds" / "round_1.json").exists()
    # the stubbed narrative was used
    assert res.narrative.executive_summary == SENTINEL_SUMMARY
    # generation was consumed: the registry contains our proposed names
    names = {c.spec.name for c in res.registry}
    assert {"month_signup", "tickets_per_spend"} & names
    # the pipeline reproduces the engineered matrix for the kept features
    mat = res.pipeline.transform(df.head(20))
    assert mat.shape[1] == len(res.kept_features)


def test_pipeline_reproduces_engineered_matrix_bit_exact(monkeypatch):
    gen = [{"name": "month_signup", "kind": "op", "op": "date_part",
            "inputs": ["signup_date"], "params": {"part": "month"}, "rationale": "cohort"}]
    monkeypatch.setattr(orch, "LLMClient", make_fake_llm(gen))
    df = make_churn_sample(n=2000, seed=11)
    res = orch.FeatureAgent(_cfg()).run(df, target="churned", task="classification")
    a = res.pipeline.transform(df)
    b = res.pipeline.transform(df.copy())
    assert a.equals(b)  # deterministic, bit-exact on a fresh frame


def test_determinism_no_llm():
    df = make_churn_sample(n=1500, seed=5)

    def run():
        cfg = _cfg(use_llm=False, output_dir=tempfile.mkdtemp(prefix="fa_det_"))
        res = orch.FeatureAgent(cfg).run(df, target="churned", task="classification")
        return [(c.spec.name, c.status, c.round_index,
                 round(c.oof_permutation_importance or 0.0, 6)) for c in res.registry]

    assert run() == run()


def test_leakage_canaries_full_pipeline(monkeypatch):
    """(a) exact target copy in forbidden_columns -> rejected_leaky;
       (b) a 98%-correlated proxy candidate -> flagged, never kept."""
    gen = [
        {"name": "copy_feature", "kind": "op", "op": "log1p", "inputs": ["account_closed"],
         "rationale": "trap: references an exact copy of the target"},
        {"name": "proxy_feature", "kind": "op", "op": "freq_encode",
         "inputs": ["churn_risk_proxy"], "rationale": "trap: derived from a 98% proxy"},
        {"name": "month_signup", "kind": "op", "op": "date_part", "inputs": ["signup_date"],
         "params": {"part": "month"}, "rationale": "legit cohort feature"},
    ]
    monkeypatch.setattr(orch, "LLMClient", make_fake_llm(gen))
    df = add_leak_columns(make_churn_sample(n=2500, seed=9), "churned")
    cfg = _cfg(forbidden_columns=["cancellation_date", "account_closed"])
    res = orch.FeatureAgent(cfg).run(df, target="churned", task="classification")

    by_name = {c.spec.name: c for c in res.registry}
    assert by_name["copy_feature"].status == "rejected_leaky"      # statically rejected
    assert by_name["proxy_feature"].status == "flagged"            # empirically flagged
    kept_names = {c.spec.name for c in res.kept_features}
    assert "proxy_feature" not in kept_names and "copy_feature" not in kept_names
