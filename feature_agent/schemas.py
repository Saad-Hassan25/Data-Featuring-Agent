"""Typed, validated data contracts for the whole pipeline.

Everything crossing a stage boundary — and everything the LLM emits or reads — is
a Pydantic model. The generator returns `FeatureSpec`s (never free-form code), the
evaluator fills `CandidateResult`s with deterministic statistics, and the LLM's
adjudication and report are validated before they can influence a decision.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

TaskType = Literal["classification", "regression"]

CandidateStatus = Literal[
    "kept",
    "pruned",
    "rejected_invalid",
    "rejected_leaky",
    "flagged",
]

SemanticType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "boolean",
    "datetime",
    "id",
    "text",
    "constant",
    "empty",
    "unknown",
]


# --------------------------------------------------------------------------- #
# Candidate feature specification (the generator's output unit)
# --------------------------------------------------------------------------- #
class FeatureSpec(BaseModel):
    """One candidate feature. Either a declarative op (preferred) or a row-wise
    expression validated by the AST whitelist (§6.2)."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,40}$")
    kind: Literal["op", "expression"]
    # kind == "op"
    op: Optional[str] = None            # e.g. "ratio", "log1p", "group_stat"
    inputs: list[str] = Field(default_factory=list)   # source columns
    params: dict = Field(default_factory=dict)        # op-specific params, schema-checked per op
    # kind == "expression"
    expression: Optional[str] = None    # row-wise numpy/pandas expression
    rationale: str = ""                 # one-sentence domain hypothesis
    hypothesis_segment: Optional[str] = None  # e.g. "matters most for enterprise tier"

    @model_validator(mode="after")
    def _check_kind(self) -> "FeatureSpec":
        if self.kind == "op":
            if not self.op:
                raise ValueError("kind='op' requires a non-empty 'op'.")
        elif self.kind == "expression":
            if not self.expression:
                raise ValueError("kind='expression' requires a non-empty 'expression'.")
        return self

    def signature(self) -> str:
        """Canonical, name-independent signature used for cross-round dedup.

        Two specs that compute the same thing under different names collapse to
        the same signature, so the generator cannot re-propose a dead end.
        """
        if self.kind == "expression":
            expr = "".join((self.expression or "").split())  # whitespace-insensitive
            return f"expr::{expr}"
        parts = sorted(str(x) for x in self.inputs)
        pk = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"op::{self.op}::in={','.join(parts)}::p={pk}"


class GenerationBatch(BaseModel):
    """The structured payload the generation LLM returns."""

    candidates: list[FeatureSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-candidate evaluation + decision record
# --------------------------------------------------------------------------- #
class CandidateResult(BaseModel):
    spec: FeatureSpec
    status: CandidateStatus
    round_index: int = 0
    oof_permutation_importance: Optional[float] = None
    shadow_percentile: Optional[float] = None       # importance vs. the shadow null distribution
    shadow_gate_passed: Optional[bool] = None
    per_fold_importance: list[float] = Field(default_factory=list)
    redundancy_cluster: Optional[int] = None
    max_abs_corr_with_existing: Optional[float] = None
    fold_stability: Optional[float] = None           # sign-consistency of importance across folds
    single_feature_score: Optional[float] = None     # AUC (clf) / |Spearman| (reg) with target
    segment: Optional[str] = None                    # segment expression, if kept within a segment
    decision_rationale: str = ""


# --------------------------------------------------------------------------- #
# Round-level summary (one per iteration)
# --------------------------------------------------------------------------- #
class RoundSummary(BaseModel):
    round_index: int
    metric_name: str
    baseline_metric: float                 # mean CV metric, original features
    augmented_metric: float                # mean CV metric, original + valid candidates + shadows
    metric_std: float                      # across-fold std of the baseline
    baseline_fold_scores: list[float] = Field(default_factory=list)
    augmented_fold_scores: list[float] = Field(default_factory=list)
    n_shadows: int = 0
    n_candidates: int = 0
    kept_count: int = 0
    confirmation_metric: Optional[float] = None      # baseline + kept-only, this round
    confirmation_passed: Optional[bool] = None
    lift_over_baseline: Optional[float] = None       # confirmation_metric - baseline_metric
    candidates: list[CandidateResult] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Data profile (compact; feeds the generation prompt and typing decisions)
# --------------------------------------------------------------------------- #
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: SemanticType
    n_unique: int = 0
    unique_rate: float = 0.0
    null_rate: float = 0.0
    # numeric sketch
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    skew: Optional[float] = None
    # categorical sketch
    top_values: list[str] = Field(default_factory=list)
    # association with target (deterministic sketch; not used for gating)
    target_association: Optional[float] = None
    sample_values: list[str] = Field(default_factory=list)


class DataProfile(BaseModel):
    n_rows: int
    n_cols: int
    target: str
    task: TaskType
    source: str = "internal"          # "internal" or "eda_agent:<path>"
    group_column: Optional[str] = None
    columns: list[ColumnProfile] = Field(default_factory=list)

    def feature_columns(self) -> list[ColumnProfile]:
        """Columns eligible as feature *inputs* (exclude target/id/constant/empty)."""
        return [
            c for c in self.columns
            if c.name != self.target and c.semantic_type not in ("id", "constant", "empty")
        ]

    def column(self, name: str) -> Optional[ColumnProfile]:
        return next((c for c in self.columns if c.name == name), None)


# --------------------------------------------------------------------------- #
# LLM adjudication of borderline candidates (§6.5)
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    """The evidence pack a borderline candidate is judged on. Read-only for the LLM."""

    name: str
    rationale: str
    hypothesis_segment: Optional[str] = None
    oof_permutation_importance: float
    shadow_percentile: float
    shadow_ceiling: float                       # 95th-pct shadow importance (the bar it must clear)
    fold_stability: float
    per_fold_importance: list[float] = Field(default_factory=list)
    correlation_neighbors: dict[str, float] = Field(default_factory=dict)  # existing col -> |rho|
    segment_metric_delta: Optional[float] = None  # lift within hypothesis_segment, if computed


class AdjudicationVerdict(BaseModel):
    name: str
    verdict: Literal["prune", "request_segment_eval"]
    segment_expression: Optional[str] = None    # required iff verdict == request_segment_eval
    reasoning: str = ""

    @model_validator(mode="after")
    def _check(self) -> "AdjudicationVerdict":
        if self.verdict == "request_segment_eval" and not self.segment_expression:
            raise ValueError("request_segment_eval requires a 'segment_expression'.")
        return self


class AdjudicationBatch(BaseModel):
    verdicts: list[AdjudicationVerdict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM-written final report narrative (§6, stage 6)
# --------------------------------------------------------------------------- #
class KeptFeatureNote(BaseModel):
    name: str
    narrative: str = Field(description="What the feature captures and why it earned its keep.")


class FeatureReportNarrative(BaseModel):
    executive_summary: str = Field(
        description="3-5 sentences: what was engineered, the lift achieved, the headline feature."
    )
    methodology_note: str = Field(
        description="How candidates were judged (marginal lift, shadow gate, leakage defense)."
    )
    kept_feature_notes: list[KeptFeatureNote] = Field(default_factory=list)
    flagged_suspicions: list[str] = Field(
        default_factory=list,
        description="Features flagged as possible leakage — routed to a human, never auto-kept.",
    )
    suggested_next_experiments: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list, description="Honest limitations of this run."
    )
