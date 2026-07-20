"""Selection + adjudication (§6.4 gates and §6.5 constrained LLM review).

Deterministic statistics decide first; the LLM only ever sees *borderline* cases,
and even then it cannot keep a feature by assertion — it can prune, or request a
concrete segment evaluation that the harness actually runs. Order matters:

  1. Empirical leakage flag — a feature that predicts the target almost alone, or
     produces an implausibly large metric swing, is *flagged* (never kept).
  2. Shadow gate + stability — split the rest into clear keeps, clear prunes, and
     borderline (§6.4 steps 5, 7).
  3. Redundancy clustering — a candidate that merely re-expresses an original
     feature, or duplicates a stronger candidate, dies here *before* verdicts
     (§6.4 step 6), so correlated candidates don't split importance and both die.
  4. LLM adjudication of the borderline set, evidence-constrained (§6.5).
  5. Confirmation ablation — baseline + kept-only must clear `min_lift`, else the
     round ships nothing (§6.4 step 8).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import FeatureAgentConfig
from .evaluate import ConfirmationResult, Evaluator, RoundEvaluation
from .executor import FeatureMaterializer
from .guards import is_empirically_suspicious
from .llm import LLMClient, LLMError
from .prompts_util import fill
from .schemas import (
    AdjudicationBatch,
    CandidateResult,
    DataProfile,
    EvidenceItem,
    FeatureSpec,
    RoundSummary,
)

_ADJ_SYSTEM = ("You are a rigorous principal data scientist. You judge borderline "
               "features strictly on the evidence provided and never fabricate support.")


# --------------------------------------------------------------------------- #
# redundancy clustering
# --------------------------------------------------------------------------- #
def _cluster_redundancy(
    specs: list[FeatureSpec], ev: RoundEvaluation, evaluator: Evaluator, config: FeatureAgentConfig,
    carried: list[FeatureSpec] | None = None,
) -> tuple[set[str], dict[str, int], dict[str, float]]:
    """Cluster candidate + original + already-kept features at |Spearman| > rho.

    Returns (redundant_pruned_names, cluster_id_by_name, max_corr_with_existing).
    A candidate clustered with an existing feature (an original column or a
    previously-kept feature — both already "free" to the model) or with a
    higher-importance candidate is pruned as redundant."""
    carried = list(carried or [])
    if not specs:
        return set(), {}, {}
    y = pd.Series(evaluator.y, index=evaluator.df.index)
    mat = FeatureMaterializer(specs=specs + carried, allowed_columns=evaluator.allowed)
    mat.fit(evaluator.df, y)
    eng = mat.transform(evaluator.df)
    base = pd.DataFrame(evaluator.X_base, columns=evaluator.base_cols, index=evaluator.df.index)
    M = pd.concat([base, eng], axis=1)
    M = M.loc[:, M.nunique(dropna=True) > 1]  # drop constants (undefined correlation)
    existing = set(evaluator.base_cols) | {s.name for s in carried}
    cand_names = [s.name for s in specs if s.name in M.columns]
    if len(M.columns) < 2 or not cand_names:
        return set(), {n: -1 for n in cand_names}, {n: 0.0 for n in cand_names}

    corr = M.corr(method="spearman").abs().fillna(0.0)
    existing_present = [c for c in M.columns if c in existing]
    max_corr = {n: (float(corr.loc[n, [c for c in existing_present if c != n]].max())
                    if len(existing_present) > (1 if n in existing_present else 0) else 0.0)
                for n in cand_names}

    labels = _hier_cluster(corr, config.redundancy_rho)
    cluster_id = {n: int(labels[list(corr.columns).index(n)]) for n in cand_names}

    def importance(col: str) -> float:
        return float("inf") if col in existing else ev.importances.get(col, 0.0)

    pruned: set[str] = set()
    for lab in set(labels):
        members = [c for c in corr.columns if labels[list(corr.columns).index(c)] == lab]
        winner = max(members, key=importance)
        for c in members:
            if c in cand_names and c != winner:
                pruned.add(c)
    return pruned, cluster_id, max_corr


def _hier_cluster(corr: pd.DataFrame, rho: float) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1.0 - corr.to_numpy()
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2.0, 0.0, None)
    if d.shape[0] < 2:
        return np.array([1])
    Z = linkage(squareform(d, checks=False), method="average")
    return fcluster(Z, t=1.0 - rho, criterion="distance")


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #
def _evidence_pack(borderline: list[FeatureSpec], ev: RoundEvaluation,
                   max_corr: dict[str, float]) -> list[EvidenceItem]:
    items = []
    for s in borderline:
        items.append(EvidenceItem(
            name=s.name, rationale=s.rationale, hypothesis_segment=s.hypothesis_segment,
            oof_permutation_importance=round(ev.importances.get(s.name, 0.0), 6),
            shadow_percentile=round(ev.shadow_percentile.get(s.name, 0.0), 1),
            shadow_ceiling=round(ev.shadow_ceiling, 6),
            fold_stability=round(ev.stability.get(s.name, 0.0), 2),
            per_fold_importance=[round(v, 6) for v in ev.per_fold.get(s.name, [])],
            correlation_neighbors={"max_abs_corr_with_original": round(max_corr.get(s.name, 0.0), 3)},
        ))
    return items


def _adjudicate(borderline: list[FeatureSpec], ev: RoundEvaluation, max_corr: dict[str, float],
                profile: DataProfile, config: FeatureAgentConfig,
                llm: LLMClient | None) -> dict[str, tuple[str, str, str]]:
    """Return name -> (verdict, segment_expression, reasoning). Prunes all when no LLM."""
    if not borderline:
        return {}
    if llm is None or not llm.available()[0]:
        return {s.name: ("prune", "", "no adjudicator available; conservative prune") for s in borderline}
    pack = _evidence_pack(borderline, ev, max_corr)
    prompt = fill("adjudicate.md", {
        "<<TASK>>": profile.task, "<<TARGET>>": profile.target,
        "<<METRIC>>": ev.metric_name, "<<SHADOW_PERCENTILE>>": str(int(config.shadow_percentile)),
        "<<EVIDENCE>>": json.dumps([i.model_dump() for i in pack], indent=2, default=str),
    })
    try:
        batch = llm.structured(stage="adjudicate", system=_ADJ_SYSTEM, user=prompt,
                               schema=AdjudicationBatch, temperature=config.adjudication_temperature)
    except LLMError:
        return {s.name: ("prune", "", "adjudicator error; conservative prune") for s in borderline}
    return {v.name: (v.verdict, v.segment_expression or "", v.reasoning) for v in batch.verdicts}


# --------------------------------------------------------------------------- #
# the selection stage
# --------------------------------------------------------------------------- #
def select_round(
    specs: list[FeatureSpec],
    ev: RoundEvaluation,
    evaluator: Evaluator,
    profile: DataProfile,
    config: FeatureAgentConfig,
    llm: LLMClient | None,
    round_index: int,
    carried_specs: list[FeatureSpec] | None = None,
) -> tuple[list[CandidateResult], list[FeatureSpec], ConfirmationResult]:
    """Run all gates over the round's *valid* candidates. `carried_specs` are
    features already kept in earlier rounds (used as 'existing' in redundancy and
    included in the confirmation ablation). Returns
    (candidate_results, new_kept_specs, confirmation)."""
    carried_specs = list(carried_specs or [])
    results: dict[str, CandidateResult] = {}
    single_scores = evaluator.single_feature_scores(specs)

    def base_result(spec: FeatureSpec, status: str, rationale: str) -> CandidateResult:
        name = spec.name
        return CandidateResult(
            spec=spec, status=status, round_index=round_index,
            oof_permutation_importance=_r(ev.importances.get(name)),
            shadow_percentile=_r(ev.shadow_percentile.get(name)),
            shadow_gate_passed=(ev.importances.get(name, 0.0) > ev.shadow_ceiling),
            per_fold_importance=[round(v, 6) for v in ev.per_fold.get(name, [])],
            fold_stability=_r(ev.stability.get(name)),
            single_feature_score=_r(single_scores.get(name)),
            decision_rationale=rationale,
        )

    # 1) empirical leakage flag ------------------------------------------- #
    remaining: list[FeatureSpec] = []
    for s in specs:
        suspicious, why = is_empirically_suspicious(
            single_scores.get(s.name), ev.importances.get(s.name), ev.baseline_std, config, profile.task)
        if suspicious:
            results[s.name] = base_result(s, "flagged", f"FLAGGED (possible leakage): {why}")
        else:
            remaining.append(s)

    # 2) shadow gate + stability ------------------------------------------ #
    clear_keep, borderline, clear_prune = [], [], []
    for s in remaining:
        imp = ev.importances.get(s.name, 0.0)
        pct = ev.shadow_percentile.get(s.name, 0.0)
        stable = ev.stability.get(s.name, 0.0) >= config.stability_min
        passed_gate = imp > ev.shadow_ceiling and pct >= config.shadow_percentile
        if passed_gate and stable:
            clear_keep.append(s)
        elif pct >= config.borderline_percentile or (passed_gate and not stable) or s.hypothesis_segment:
            borderline.append(s)
        else:
            clear_prune.append(s)
    for s in clear_prune:
        results[s.name] = base_result(
            s, "pruned", f"below shadow gate (percentile {ev.shadow_percentile.get(s.name, 0):.0f} "
                         f"< {config.shadow_percentile:.0f}).")

    # 3) redundancy clustering (before verdicts) -------------------------- #
    considered = clear_keep + borderline
    redundant, cluster_id, max_corr = _cluster_redundancy(
        considered, ev, evaluator, config, carried=carried_specs)
    clear_keep = [s for s in clear_keep if s.name not in redundant]
    borderline = [s for s in borderline if s.name not in redundant]
    for name in redundant:
        s = next(x for x in considered if x.name == name)
        r = base_result(s, "pruned",
                        f"redundant — clusters with an existing/stronger feature "
                        f"(|Spearman| with original ≈ {max_corr.get(name, 0):.2f}).")
        r.redundancy_cluster = cluster_id.get(name)
        r.max_abs_corr_with_existing = _r(max_corr.get(name))
        results[name] = r

    # 4) adjudication of borderline --------------------------------------- #
    verdicts = _adjudicate(borderline, ev, max_corr, profile, config, llm)
    kept_specs: list[FeatureSpec] = list(clear_keep)
    for s in clear_keep:
        r = base_result(s, "kept", "cleared the shadow gate with stable across-fold importance.")
        r.redundancy_cluster = cluster_id.get(s.name)
        r.max_abs_corr_with_existing = _r(max_corr.get(s.name))
        results[s.name] = r
    for s in borderline:
        verdict, seg_expr, reasoning = verdicts.get(
            s.name, ("prune", "", "no verdict returned; conservative prune"))
        if verdict == "request_segment_eval" and seg_expr:
            seg = evaluator.segment_eval(s, seg_expr)
            if seg.passed:
                r = base_result(s, "kept",
                                f"kept within segment `{seg_expr}`: {reasoning} ({seg.note}).")
                r.segment = seg_expr
                kept_specs.append(s)
            else:
                r = base_result(s, "pruned",
                                f"segment `{seg_expr}` did not clear the gate: {seg.note}.")
        else:
            r = base_result(s, "pruned", f"adjudicator pruned: {reasoning}")
        r.redundancy_cluster = cluster_id.get(s.name)
        r.max_abs_corr_with_existing = _r(max_corr.get(s.name))
        results[s.name] = r

    # 5) confirmation ablation (on the full cumulative kept set) ---------- #
    full_kept = carried_specs + kept_specs
    confirmation = evaluator.confirmation(full_kept)
    if kept_specs and not confirmation.passed:
        for s in kept_specs:
            results[s.name].status = "pruned"
            results[s.name].decision_rationale += (
                f" | Reverted: cumulative confirmation ablation lift {confirmation.lift:.4f} "
                f"< min_lift {confirmation.min_lift:.4f}; round ships zero new keeps.")
        kept_specs = []

    ordered = [results[s.name] for s in specs if s.name in results]
    return ordered, kept_specs, confirmation


def _r(x, ndigits: int = 6):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, ndigits)
