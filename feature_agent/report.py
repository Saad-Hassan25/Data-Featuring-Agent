"""Reporting + reproducible artifacts (§3 stage 6, §7).

Three outputs:
  * **feature_registry.json** — every candidate ever proposed (kept, pruned,
    rejected, flagged) with its spec, scores, decision rationale, and round. This
    is the documentation the manual process never produces: "we tried X and it
    failed the shadow gate" prevents the next run from re-proposing dead ends.
  * **pipeline.joblib** — an sklearn Pipeline mapping a raw DataFrame to the kept
    engineered matrix, with fitted transforms refit on the full training frame.
  * **report.md** — an LLM-written narrative (deterministic fallback) grounded in
    the deterministic evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import FeatureAgentConfig
from .executor import FeatureMaterializer
from .llm import LLMClient, LLMError
from .prompts_util import fill
from .schemas import (
    CandidateResult,
    DataProfile,
    FeatureReportNarrative,
    KeptFeatureNote,
    RoundSummary,
)

_REPORT_SYSTEM = ("You are a principal data scientist. Interpret pre-computed statistics "
                  "faithfully; never invent numbers. Be direct and honest about limits.")


# --------------------------------------------------------------------------- #
# reproducible artifacts
# --------------------------------------------------------------------------- #
def export_pipeline(kept_specs, allowed_columns: list[str], df: pd.DataFrame,
                    y: pd.Series) -> Pipeline:
    """Fit the export pipeline on the full training frame (fold-fitted transforms
    are refit here on all training rows, per §7)."""
    pipe = Pipeline([("features", FeatureMaterializer(specs=list(kept_specs),
                                                      allowed_columns=list(allowed_columns)))])
    pipe.fit(df, y)
    return pipe


def write_registry(results: list[CandidateResult], path: Path) -> None:
    payload = [r.model_dump(exclude_none=True) for r in results]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_round(summary: RoundSummary, path: Path) -> None:
    path.write_text(json.dumps(summary.model_dump(exclude_none=True), indent=2, default=str),
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# narrative
# --------------------------------------------------------------------------- #
@dataclass
class ReportContext:
    profile: DataProfile
    metric_name: str
    baseline_metric: float
    final_metric: float
    lift: float
    min_lift: float
    confirmation_passed: bool
    kept: list[CandidateResult] = field(default_factory=list)
    flagged: list[CandidateResult] = field(default_factory=list)
    notable_prunes: list[CandidateResult] = field(default_factory=list)
    rounds: list[RoundSummary] = field(default_factory=list)


def _summary_stats(ctx: ReportContext) -> str:
    return (f"baseline {ctx.metric_name}={ctx.baseline_metric:.4f}; "
            f"confirmation (baseline + {len(ctx.kept)} kept)={ctx.final_metric:.4f}; "
            f"lift={ctx.lift:+.4f} (min required {ctx.min_lift:.4f}, "
            f"{'passed' if ctx.confirmation_passed else 'NOT passed'}).")


def _kept_block(ctx: ReportContext) -> str:
    if not ctx.kept:
        return "(none kept this run)"
    lines = []
    for c in ctx.kept:
        seg = f", segment=`{c.segment}`" if c.segment else ""
        lines.append(
            f"- `{c.spec.name}` [{c.spec.kind}:{c.spec.op or 'expr'}] — "
            f"importance={c.oof_permutation_importance:.5f}, "
            f"shadow_pct={c.shadow_percentile:.0f}, stability={c.fold_stability:.2f}{seg}. "
            f"Hypothesis: {c.spec.rationale}")
    return "\n".join(lines)


def _flagged_block(ctx: ReportContext) -> str:
    if not ctx.flagged:
        return "(none)"
    return "\n".join(f"- `{c.spec.name}` — {c.decision_rationale}" for c in ctx.flagged)


def _pruned_block(ctx: ReportContext) -> str:
    if not ctx.notable_prunes:
        return "(none notable)"
    return "\n".join(f"- `{c.spec.name}` ({c.status}) — {c.decision_rationale}"
                     for c in ctx.notable_prunes[:15])


def _rounds_block(ctx: ReportContext) -> str:
    lines = []
    for r in ctx.rounds:
        lines.append(f"- Round {r.round_index}: {r.n_candidates} candidates, {r.kept_count} kept; "
                     f"baseline={r.baseline_metric:.4f}, augmented={r.augmented_metric:.4f}, "
                     f"confirmation lift={_fmt(r.lift_over_baseline)}.")
    return "\n".join(lines) or "(no rounds)"


def deterministic_report(ctx: ReportContext) -> FeatureReportNarrative:
    verb = "improved" if ctx.lift > 0 else "did not improve"
    summary = (
        f"Engineered and tested candidate features for `{ctx.profile.target}` "
        f"({ctx.profile.task}). {len(ctx.kept)} feature(s) survived a leakage-safe, "
        f"marginal-lift selection. Baseline {ctx.metric_name} {ctx.baseline_metric:.4f}; "
        f"with kept features {ctx.final_metric:.4f} ({ctx.lift:+.4f}) — confirmation "
        f"ablation {'passed' if ctx.confirmation_passed else 'did not pass'}. "
        f"The kept set {verb} the baseline.")
    return FeatureReportNarrative(
        executive_summary=summary,
        methodology_note=(
            "Candidates were judged by paired baseline-vs-augmented cross-validation "
            "on identical folds. Attribution used out-of-fold permutation importance; "
            "significance was set by a shadow (noise) feature gate, not a fixed threshold. "
            "Fitted transforms were refit inside each fold to prevent leakage."),
        kept_feature_notes=[KeptFeatureNote(
            name=c.spec.name,
            narrative=f"{c.spec.rationale} (OOF permutation importance {c.oof_permutation_importance:.5f}, "
                      f"shadow percentile {c.shadow_percentile:.0f}).") for c in ctx.kept],
        flagged_suspicions=[f"{c.spec.name}: {c.decision_rationale}" for c in ctx.flagged],
        suggested_next_experiments=_default_next_steps(ctx),
        caveats=[
            "Narrative generated by rules (no LLM). All statistics are computed deterministically.",
            "Marginal lift is evidence, not proof of causation; validate kept features on a holdout.",
            "SHAP direction/segment narratives were not computed in this run.",
        ],
    )


def _default_next_steps(ctx: ReportContext) -> list[str]:
    steps = []
    kept_ops = {c.spec.op for c in ctx.kept if c.spec.op}
    if "ratio" in kept_ops:
        steps.append("Ratios survived — try more spend/usage-normalized rates.")
    if "group_stat" in kept_ops:
        steps.append("Group-relative stats helped — explore deviations from the group stat.")
    if ctx.flagged:
        steps.append("Have a human confirm the flagged features are not leakage before any reuse.")
    if not steps:
        steps.append("Broaden the candidate vocabulary or add domain context to the next run.")
    return steps


def llm_report(llm: LLMClient | None, ctx: ReportContext,
               config: FeatureAgentConfig) -> tuple[FeatureReportNarrative, str]:
    """LLM-written narrative with graceful fallback. Uses the (cheaper) fallback
    model for the report stage when one is configured."""
    if llm is None or not llm.available()[0]:
        return deterministic_report(ctx), "deterministic (no LLM)"
    prompt = fill("report.md", {
        "<<TASK>>": ctx.profile.task, "<<TARGET>>": ctx.profile.target,
        "<<METRIC>>": ctx.metric_name, "<<SUMMARY_STATS>>": _summary_stats(ctx),
        "<<KEPT>>": _kept_block(ctx), "<<FLAGGED>>": _flagged_block(ctx),
        "<<PRUNED>>": _pruned_block(ctx), "<<ROUNDS>>": _rounds_block(ctx),
    })
    try:
        narrative = llm.structured(
            stage="report", system=_REPORT_SYSTEM, user=prompt,
            schema=FeatureReportNarrative, temperature=config.adjudication_temperature,
            model=config.fallback_model or config.model)
        return narrative, f"LLM ({config.fallback_model or config.model})"
    except LLMError as exc:
        nar = deterministic_report(ctx)
        nar.caveats.insert(0, f"LLM narrative unavailable: {exc}")
        return nar, f"deterministic (LLM error: {exc})"


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown(narrative: FeatureReportNarrative, ctx: ReportContext,
                    source: str, generated_at: str = "") -> str:
    L: list[str] = []
    badge = "🟢 lift confirmed" if ctx.confirmation_passed else "⚪ no confirmed lift"
    L += ["# Feature Engineering & Selection Report", ""]
    L.append(f"**Target:** `{ctx.profile.target}`  ·  **Task:** {ctx.profile.task}  ·  "
             f"**Metric:** {ctx.metric_name}  ·  **Outcome:** {badge}  ")
    if generated_at:
        L.append(f"**Generated:** {generated_at}  ")
    L.append(f"**Narrative source:** {source}")
    L += ["", "## Executive summary", "", narrative.executive_summary, ""]
    L += ["## Result", "",
          f"- **Baseline {ctx.metric_name}:** {ctx.baseline_metric:.4f}",
          f"- **With {len(ctx.kept)} kept feature(s):** {ctx.final_metric:.4f} "
          f"(**{ctx.lift:+.4f}**, min required {ctx.min_lift:.4f})",
          f"- **Confirmation ablation:** {'passed ✅' if ctx.confirmation_passed else 'not passed'}",
          ""]
    L += ["## Methodology", "", narrative.methodology_note, ""]

    L += ["## Kept features", ""]
    if ctx.kept:
        L += ["| Feature | Kind | OOF importance | Shadow pct | Stability | Segment |",
              "|---|---|---|---|---|---|"]
        for c in ctx.kept:
            L.append(f"| `{c.spec.name}` | {c.spec.kind}:{c.spec.op or 'expr'} | "
                     f"{_fmt(c.oof_permutation_importance)} | {_fmt(c.shadow_percentile)} | "
                     f"{_fmt(c.fold_stability)} | {c.segment or '—'} |")
        L.append("")
        for note in narrative.kept_feature_notes:
            L.append(f"- **`{note.name}`** — {note.narrative}")
        L.append("")
    else:
        L += ["_No features were kept — the round(s) did not clear the confirmation gate._", ""]

    if narrative.flagged_suspicions:
        L += ["## ⚠️ Flagged as possible leakage (NOT kept — human review)", ""]
        L += [f"- {s}" for s in narrative.flagged_suspicions] + [""]

    L += ["## Notable prunes & rejections", "", _pruned_block(ctx), ""]

    if narrative.suggested_next_experiments:
        L += ["## Suggested next experiments", ""]
        L += [f"{i}. {s}" for i, s in enumerate(narrative.suggested_next_experiments, 1)] + [""]

    L += ["## Round-by-round", "", _rounds_block(ctx), ""]

    if narrative.caveats:
        L += ["## Caveats & limitations", ""]
        L += [f"- {c}" for c in narrative.caveats] + [""]

    L += ["---", "*Generated by the Feature Agent. Statistics are computed deterministically "
          "via a leakage-safe CV harness; the narrative is an LLM reasoning over those statistics.*"]
    return "\n".join(L)


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)
