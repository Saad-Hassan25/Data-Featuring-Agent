"""Agentic Feature Engineering & Selection agent.

A principled pipeline that proposes, materializes, cross-validates, prunes, and
documents candidate features:

  * an LLM proposes feature *specs* from a fixed op vocabulary (never raw code),
  * a safe executor materializes them (declarative ops + AST-whitelisted expressions),
  * a leakage-safe CV harness measures each candidate's *marginal lift* over the
    baseline with out-of-fold permutation importance and a shadow-feature gate,
  * deterministic gates prune, the LLM adjudicates only borderline cases on evidence,
  * every run emits a feature registry, a serialized sklearn pipeline, and a report.

Public API:
    from feature_agent import FeatureAgent, FeatureAgentConfig
    result = FeatureAgent(FeatureAgentConfig(model="<openrouter-model-id>")).run(
        df, target="churned", task="classification", domain_context="...")
    result.kept_features   # list[CandidateResult]
    result.pipeline        # sklearn Pipeline: raw df -> engineered matrix
    result.report_path     # report.md
"""

from __future__ import annotations

from .config import FeatureAgentConfig
from .orchestrator import FeatureAgent, FeatureAgentResult
from .schemas import (
    CandidateResult,
    DataProfile,
    FeatureReportNarrative,
    FeatureSpec,
    RoundSummary,
)

__all__ = [
    "FeatureAgent",
    "FeatureAgentConfig",
    "FeatureAgentResult",
    "FeatureSpec",
    "CandidateResult",
    "RoundSummary",
    "DataProfile",
    "FeatureReportNarrative",
]

__version__ = "0.1.0"
