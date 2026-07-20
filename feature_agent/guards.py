"""Leakage defense and candidate validation (§6.3).

Three layers of leakage defense live in this codebase; this module owns two of
them and the exception types the others raise:

  1. **Static (here).** Before a candidate is ever executed we reject: references
     to the target or user-declared post-outcome columns; unknown/ineligible input
     columns; name collisions with existing columns; and fitted group ops keyed on
     a near-unique column (row identity, which leaks the target trivially).
  2. **Structural (evaluate.py).** Fitted transforms are refit inside every CV
     fold on training rows only — never on the full frame.
  3. **Empirical (here + select.py).** After evaluation, suspicion heuristics
     *flag* rather than silently keep a candidate that predicts the target almost
     perfectly on its own or produces an implausible metric jump.

A leaky feature looks spectacular in evaluation and poisons the downstream model,
so the default posture is: reject early, and when in doubt flag for a human.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureAgentConfig
from .schemas import DataProfile, FeatureSpec


class InvalidCandidate(Exception):
    """A candidate cannot be built or executed safely (structured, machine-readable)."""


class LeakageError(InvalidCandidate):
    """A candidate references the target / a forbidden column, or leaks by construction."""


# --------------------------------------------------------------------------- #
# Static validation (pre-execution)
# --------------------------------------------------------------------------- #
def allowed_columns(profile: DataProfile, config: FeatureAgentConfig) -> set[str]:
    """Columns a candidate may reference: everything except the target, the group
    key, user-declared forbidden/post-outcome columns, and declared id columns.

    Excluding the target here means a formula that references it fails *validation*
    rather than needing to be caught after the fact (§6.2)."""
    blocked = {profile.target}
    blocked.update(config.forbidden_columns)
    blocked.update(config.id_columns)
    if config.group_column:
        blocked.add(config.group_column)
    return {c.name for c in profile.columns if c.name not in blocked}


def _near_unique(profile: DataProfile, column: str, config: FeatureAgentConfig) -> bool:
    cp = profile.column(column)
    if cp is None:
        return False
    # cardinality > 50% of rows -> effectively row identity
    return cp.unique_rate > 0.5 or cp.semantic_type == "id"


# Fitted ops that aggregate a target-correlated statistic by a key: dangerous if
# the key is near-unique (each group is ~one row, so the stat memorizes the row).
_KEY_GROUPED_OPS = {"group_stat", "target_encode", "count_encode", "freq_encode"}


def validate_spec_static(
    spec: FeatureSpec,
    profile: DataProfile,
    config: FeatureAgentConfig,
    existing_names: set[str],
) -> None:
    """Raise InvalidCandidate / LeakageError if the spec is unsafe or malformed.

    Deterministic and pure; the structured reason is fed back into the next
    generation round so the model learns the boundary instead of repeating it.
    """
    allowed = allowed_columns(profile, config)
    blocked = ({profile.target} | set(config.forbidden_columns)
               | set(config.id_columns) | ({config.group_column} if config.group_column else set()))

    # name collision with a real column or an already-accepted candidate
    if spec.name in {c.name for c in profile.columns}:
        raise InvalidCandidate(f"name '{spec.name}' collides with an existing column.")
    if spec.name in existing_names:
        raise InvalidCandidate(f"name '{spec.name}' duplicates another candidate this run.")

    if spec.kind == "op":
        if not spec.inputs:
            raise InvalidCandidate(f"op '{spec.op}' requires at least one input column.")
        for col in spec.inputs:
            if col in blocked:
                raise LeakageError(
                    f"input '{col}' references the target or a forbidden/post-outcome column."
                )
            if col not in allowed:
                raise InvalidCandidate(f"unknown or ineligible input column '{col}'.")
        # a fitted group op keyed on a near-unique column memorizes row identity
        if spec.op in _KEY_GROUPED_OPS:
            key = _grouping_key(spec)
            if key and _near_unique(profile, key, config):
                raise LeakageError(
                    f"op '{spec.op}' is grouped on near-unique key '{key}' "
                    f"(cardinality > 50% of rows) — this leaks row identity."
                )
        if spec.op == "target_encode" and not config.enable_target_encode:
            raise InvalidCandidate(
                "target_encode is disabled by default (enable_target_encode=False)."
            )
    else:  # expression — column checks happen in the AST validator (executor);
        # here we only pre-screen for obvious target/forbidden references so the
        # rejection reason is precise. The AST validator is the authoritative gate.
        pass


def _grouping_key(spec: FeatureSpec) -> str | None:
    """The categorical key a fitted group op aggregates over."""
    if "by" in spec.params:
        return str(spec.params["by"])
    if spec.op in {"count_encode", "freq_encode", "target_encode"}:
        return spec.inputs[0] if spec.inputs else None
    if spec.op == "group_stat":
        # group_stat groups a numeric 'value' by a categorical 'by'; inputs are
        # [by, value] when params.by is absent.
        return spec.inputs[0] if spec.inputs else None
    return None


# --------------------------------------------------------------------------- #
# Empirical suspicion heuristics (post-evaluation, §6.3 layer 3)
# --------------------------------------------------------------------------- #
def single_feature_score(x: pd.Series, y: pd.Series, task: str) -> float | None:
    """Standalone predictive strength of a single materialized feature vs. target.

    Classification: AUC of the feature as a raw score (max of the two sign
    orientations). Regression: |Spearman| with the target. Values near 1.0 mean
    the feature nearly *is* the target — the signature of a leak.
    """
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce").to_numpy(),
                      "y": y.to_numpy()}).dropna()
    if d.shape[0] < 20 or d["x"].nunique() < 2:
        return None
    if task == "classification":
        yb = d["y"]
        if yb.nunique() != 2:
            return None
        try:
            from sklearn.metrics import roc_auc_score
            # binarize target to {0,1} by its sorted classes
            classes = sorted(yb.unique())
            y01 = (yb == classes[-1]).astype(int)
            auc = roc_auc_score(y01, d["x"])
            return float(max(auc, 1 - auc))
        except Exception:
            return None
    # regression
    from scipy import stats
    rho = stats.spearmanr(d["x"], d["y"]).statistic
    return None if rho is None or np.isnan(rho) else float(abs(rho))


def is_empirically_suspicious(
    single_score: float | None,
    metric_jump: float | None,
    baseline_std: float,
    config: FeatureAgentConfig,
    task: str,
) -> tuple[bool, str]:
    """Decide whether a candidate should be *flagged* (not kept) as possible leakage.

    The single-feature score (AUC / |Spearman| with the target) is the primary
    signal: a value near 1 means the feature nearly *is* the target. The
    implausible-jump check is deliberately conjoined with a strong single-feature
    score — a large marginal lift *alone* is what a genuinely good, base-invisible
    feature produces (e.g. a date-derived signal the model otherwise can't see), so
    flagging on the jump alone would reject exactly the features we want to keep."""
    kind = "AUC" if task == "classification" else "|Spearman|"
    thr = config.leakage_single_auc if task == "classification" else config.leakage_single_corr
    if single_score is not None and single_score >= thr:
        return True, (f"single-feature {kind}={single_score:.3f} ≥ {thr:.2f} with the target "
                      f"— predicts the target almost alone; probable leakage.")
    if (metric_jump is not None and baseline_std > 0
            and metric_jump > config.implausible_jump_std_mult * baseline_std
            and single_score is not None and single_score >= thr - 0.05):
        return True, (f"adding this feature moved the CV metric by {metric_jump:.4f} "
                      f"(> {config.implausible_jump_std_mult:g}× the baseline fold std "
                      f"{baseline_std:.4f}) while also tracking the target on its own "
                      f"({kind}={single_score:.3f}) — implausibly strong; probable leakage.")
    return False, ""
