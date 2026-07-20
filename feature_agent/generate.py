"""Candidate generation (§3 stage 2).

The LLM proposes `FeatureSpec`s from the op vocabulary, grounded in the data
profile, the domain context, and structured feedback from prior rounds. Output is
schema-validated (in `llm.py`) and then deduplicated — by canonical signature —
against everything proposed so far, so a dead end is never re-proposed.

A deterministic generator provides the `--no-llm` fallback and a network-free
engine for tests: it proposes the sensible default shapes a practitioner starts
with (ratios and products among the most target-associated numerics, log/sqrt of
skewed columns, group-relative stats, frequency encodings, missingness flags, date
parts). This is what lets the recovery benchmark run without a live model.
"""

from __future__ import annotations

import re

from .config import FeatureAgentConfig
from .guards import allowed_columns as _allowed_columns
from .llm import LLMClient, LLMError
from .ops import describe_vocabulary
from .prompts_util import fill
from .schemas import ColumnProfile, DataProfile, FeatureSpec, GenerationBatch, RoundSummary

_GEN_SYSTEM = ("You are a rigorous, practical principal data scientist. You propose "
               "feature specifications from a fixed vocabulary; you never write code.")


def eligible_columns(profile: DataProfile, config: FeatureAgentConfig) -> list[ColumnProfile]:
    """Column profiles usable as feature inputs: feature columns intersected with
    the leakage-allowed set (excludes target, forbidden/post-outcome, group, ids)."""
    allowed = _allowed_columns(profile, config)
    return [c for c in profile.feature_columns() if c.name in allowed]


# --------------------------------------------------------------------------- #
# prompt rendering
# --------------------------------------------------------------------------- #
def _render_columns(columns: list[ColumnProfile]) -> str:
    lines = []
    for c in columns:
        bits = [f"`{c.name}`", f"type={c.semantic_type}", f"card={c.n_unique}",
                f"null={c.null_rate:.0%}"]
        if c.target_association is not None:
            bits.append(f"|assoc|~{c.target_association:.2f}")
        if c.sample_values:
            bits.append("e.g. " + ", ".join(c.sample_values[:3]))
        lines.append("- " + "  ".join(bits))
    return "\n".join(lines) or "(no eligible columns)"


def _render_profile(profile: DataProfile) -> str:
    return (f"{profile.n_rows:,} rows × {profile.n_cols} columns. "
            f"Task: {profile.task}. Profile source: {profile.source}.")


def _render_vocab(config: FeatureAgentConfig) -> str:
    lines = []
    for op in describe_vocabulary(config.enable_target_encode):
        params = ("; ".join(f"{k}: {v}" for k, v in op["params"].items()) or "none")
        lines.append(f"- **{op['op']}** ({op['fold_safety']}) — {op['description']}\n"
                     f"    inputs: {op['inputs']}; params: {params}")
    return "\n".join(lines)


def _render_feedback(history: list[RoundSummary]) -> str:
    if not history:
        return "(first round — no feedback yet)"
    last = history[-1]
    out = [f"After round {last.round_index}: baseline {last.metric_name}="
           f"{last.baseline_metric:.4f}, augmented={last.augmented_metric:.4f}, "
           f"{last.kept_count} kept."]
    survivors = [c for c in last.candidates if c.status == "kept"]
    if survivors:
        survivors.sort(key=lambda c: c.oof_permutation_importance or 0, reverse=True)
        out.append("SURVIVED (double down on these shapes): " + ", ".join(
            f"{c.spec.name}[{c.spec.kind}:{c.spec.op or 'expr'}]" for c in survivors[:8]))
    pruned = [c for c in last.candidates if c.status == "pruned"]
    if pruned:
        out.append("PRUNED (weak marginal signal — avoid these shapes): " + ", ".join(
            f"{c.spec.name}" for c in pruned[:8]))
    rejected = [c for c in last.candidates if c.status in ("rejected_invalid", "rejected_leaky")]
    if rejected:
        out.append("REJECTED (do not repeat): " + "; ".join(
            f"{c.spec.name} — {c.decision_rationale}" for c in rejected[:8]))
    flagged = [c for c in last.candidates if c.status == "flagged"]
    if flagged:
        out.append("FLAGGED as possible leakage (a proxy for the target — do not propose "
                   "near-target features): " + ", ".join(c.spec.name for c in flagged))
    return "\n".join(out)


def build_generation_prompt(profile: DataProfile, config: FeatureAgentConfig,
                            domain_context: str, history: list[RoundSummary]) -> str:
    return fill("generate.md", {
        "<<TASK>>": profile.task,
        "<<TARGET>>": profile.target,
        "<<DOMAIN_CONTEXT>>": domain_context or "(none provided)",
        "<<N_CANDIDATES>>": str(config.n_candidates_per_round),
        "<<PROFILE>>": _render_profile(profile),
        "<<OP_VOCAB>>": _render_vocab(config),
        "<<ALLOWED_COLUMNS>>": _render_columns(eligible_columns(profile, config)),
        "<<FEEDBACK>>": _render_feedback(history),
    })


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def dedup(specs: list[FeatureSpec], seen: set[str]) -> list[FeatureSpec]:
    out, local_sig, local_names = [], set(), set()
    for s in specs:
        sig = s.signature()
        if sig in seen or sig in local_sig or s.name in local_names:
            continue
        local_sig.add(sig)
        local_names.add(s.name)
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# LLM generation
# --------------------------------------------------------------------------- #
def generate_candidates(
    profile: DataProfile, config: FeatureAgentConfig, domain_context: str,
    llm: LLMClient | None, history: list[RoundSummary], seen: set[str],
) -> list[FeatureSpec]:
    specs: list[FeatureSpec]
    if llm is not None and llm.available()[0]:
        prompt = build_generation_prompt(profile, config, domain_context, history)
        try:
            batch = llm.structured(
                stage="generate", system=_GEN_SYSTEM, user=prompt,
                schema=GenerationBatch, temperature=config.generation_temperature,
            )
            specs = batch.candidates
        except LLMError:
            specs = deterministic_candidates(profile, config)
    else:
        specs = deterministic_candidates(profile, config)
    return dedup(specs, seen)


# --------------------------------------------------------------------------- #
# deterministic generator (fallback + tests)
# --------------------------------------------------------------------------- #
def _mk_name(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts).lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if not raw or not raw[0].isalpha():
        raw = "f_" + raw
    raw = raw[:41]
    while len(raw) < 3:
        raw += "_x"
    return raw


def deterministic_candidates(profile: DataProfile, config: FeatureAgentConfig) -> list[FeatureSpec]:
    """Practitioner-default candidates: ratios/products among top numerics, log/sqrt
    of skewed columns, group stats, frequency encodings, missingness flags, dates."""
    cols = eligible_columns(profile, config)
    numerics = [c for c in cols if c.semantic_type in ("numeric_continuous", "numeric_discrete")]
    numerics.sort(key=lambda c: (c.target_association or 0.0), reverse=True)
    cats = [c for c in cols if c.semantic_type in ("categorical", "boolean")
            and 2 <= c.n_unique <= 200 and c.unique_rate <= 0.5]
    dates = [c for c in cols if c.semantic_type == "datetime"]
    missing = [c for c in cols if 0.0 < c.null_rate < 1.0]

    specs: list[FeatureSpec] = []
    seen_names: set[str] = set()

    def add(spec: FeatureSpec) -> None:
        if spec.name in seen_names:
            return
        seen_names.add(spec.name)
        specs.append(spec)

    top_num = numerics[:6]
    # ratios (both directions) + products among the most target-associated numerics
    for i, a in enumerate(top_num):
        for b in top_num:
            if a.name == b.name:
                continue
            add(FeatureSpec(name=_mk_name("ratio", a.name, b.name), kind="op", op="ratio",
                            inputs=[a.name, b.name],
                            rationale=f"rate of {a.name} per unit {b.name}"))
        for b in top_num[i + 1:]:
            add(FeatureSpec(name=_mk_name("prod", a.name, b.name), kind="op", op="product",
                            inputs=[a.name, b.name],
                            rationale=f"interaction of {a.name} and {b.name}"))
    # log/sqrt of skewed numerics
    for c in numerics:
        if c.skew is not None and abs(c.skew) >= 1.0:
            add(FeatureSpec(name=_mk_name("log1p", c.name), kind="op", op="log1p",
                            inputs=[c.name], rationale=f"tame the skew of {c.name}"))
            add(FeatureSpec(name=_mk_name("sqrt", c.name), kind="op", op="sqrt",
                            inputs=[c.name], rationale=f"variance-stabilize {c.name}"))
    # group-relative stats: top numeric within each categorical
    for cat in cats[:3]:
        for num in top_num[:2]:
            add(FeatureSpec(name=_mk_name("grpmean", num.name, cat.name), kind="op", op="group_stat",
                            inputs=[cat.name, num.name], params={"stat": "mean"},
                            rationale=f"typical {num.name} within each {cat.name}"))
    # frequency encodings for categoricals
    for cat in cats[:4]:
        add(FeatureSpec(name=_mk_name("freq", cat.name), kind="op", op="freq_encode",
                        inputs=[cat.name], rationale=f"prevalence of each {cat.name} level"))
    # missingness flags
    for c in missing[:4]:
        add(FeatureSpec(name=_mk_name("ismiss", c.name), kind="op", op="is_missing",
                        inputs=[c.name], rationale=f"whether {c.name} is recorded"))
    # date parts (base excludes raw dates, so these often expose genuine signal)
    for d in dates[:2]:
        for part in ("month", "dayofweek", "is_weekend"):
            add(FeatureSpec(name=_mk_name(part, d.name), kind="op", op="date_part",
                            inputs=[d.name], params={"part": part},
                            rationale=f"{part} effect in {d.name}"))
    return specs
