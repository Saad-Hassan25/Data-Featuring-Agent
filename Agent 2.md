# Agent 2 — Agentic Feature Engineering & Selection

**Status:** Design approved for implementation · **LLM provider:** OpenRouter (all model calls) · **Companion:** consumes the data profile produced by `eda_agent` (read-only dependency; falls back to internal profiling if absent)

---

## 1. Problem Statement

**What it replaces.** The manual loop every ML project runs: brainstorming interaction/transformation features, writing the pandas code, evaluating each candidate against a baseline, pruning the ones that don't contribute, and documenting what survived and why. This loop is slow, inconsistent across practitioners, rarely documented, and almost never leakage-audited.

**What the agent does.** Given a dataset, a target, and a task type, the agent:

1. Profiles the data (or ingests the `eda_agent` profile).
2. Proposes candidate features via an LLM, grounded in the data profile and domain context.
3. Materializes candidates through a **safe, validated executor** — never raw code execution.
4. Evaluates candidates with a **leakage-safe CV protocol** measuring *marginal lift over the baseline*, not standalone importance.
5. Prunes using a **statistically principled gate** (shadow-feature benchmark + redundancy clustering), then lets the LLM adjudicate borderline cases with evidence.
6. Iterates: feeds results back to the generator for further rounds until lift plateaus or budget is exhausted.
7. Emits reproducible artifacts: a feature registry, a serialized transform pipeline, and a written rationale report.

**Non-goals (v1).** Deep-learning feature learning, time-series-specific features (lags/windows), automated target transformation, multi-table/relational feature synthesis (Featuretools territory), and hyperparameter tuning of the downstream model. These are listed in §13 as explicit extensions.

---

## 2. Lessons from the Prototype (what this plan fixes)

The prototype validated the core idea but has six defects that would sink it in production. Each drives a design decision below.

| # | Prototype defect | Consequence | Fix (section) |
|---|---|---|---|
| 1 | Direct `OpenAI()` client, hardcoded `gpt-4o-mini` | Wrong provider; no model flexibility | OpenRouter client wrapper, model in config (§6.1) |
| 2 | `df.eval()` on raw LLM formula strings | Arbitrary-expression execution; fails on groupby/string/date ops; silent skips | Two-tier safe executor: declarative op specs + AST-whitelisted expressions (§6.2) |
| 3 | Model trained on candidate features **only**, in-sample, no CV | Importance shares among candidates are meaningless; no evidence any candidate beats the baseline | Baseline-vs-augmented CV protocol with out-of-fold attribution (§6.4) |
| 4 | Fixed importance threshold (0.01 of normalized split gain) | Threshold meaning changes with candidate count; split-gain is biased toward high-cardinality features | Shadow (noise) feature gate — Boruta-style adaptive threshold (§6.4) |
| 5 | No leakage defense | LLM can propose formulas referencing the target or near-target proxies; fitted transforms (target encoding, group stats) computed on full data leak across folds | Static leakage validator + fold-fitted transformers + leakage canary tests (§6.3) |
| 6 | Blanket `fillna(0)`, no categorical handling, classification only | Wrong on skewed/categorical data; half the use cases unsupported | Typed preprocessing, native LightGBM categoricals, regression + classification (§6.4) |

---

## 3. Architecture

One agent, six pipeline stages, orchestrated by a deterministic controller. The LLM is called at three points (generation, adjudication, reporting); everything else is deterministic code so runs are reproducible and cheap.

```
                          ┌──────────────────────────────────────────────┐
                          │                ORCHESTRATOR                  │
                          │   (round loop, budgets, run manifest)        │
                          └──────────────────────────────────────────────┘
                                              │
   ┌──────────┐    ┌────────────┐    ┌────────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────┐
   │ 1.       │    │ 2.         │    │ 3.             │    │ 4.         │    │ 5.          │    │ 6.       │
   │ PROFILE  │───▶│ GENERATE   │───▶│ VALIDATE +     │───▶│ EVALUATE   │───▶│ SELECT +    │───▶│ REPORT   │
   │          │    │ (LLM)      │    │ MATERIALIZE    │    │ (CV)       │    │ ADJUDICATE  │    │ (LLM)    │
   │ schema,  │    │ candidate  │    │ static checks, │    │ baseline Δ,│    │ shadow gate,│    │ registry,│
   │ stats,   │    │ specs via  │    │ safe executor, │    │ OOF perm.  │    │ redundancy, │    │ pipeline,│
   │ EDA hand-│    │ OpenRouter │    │ leakage guard  │    │ importance,│    │ LLM (LLM)   │    │ report.md│
   │ off      │    │            │    │                │    │ shadows    │    │ borderline  │    │          │
   └──────────┘    └────────────┘    └────────────────┘    └────────────┘    └─────────────┘    └──────────┘
                         ▲                                                          │
                         └──────────────── feedback: survivors, failures, ──────────┘
                                           rationales → next round (≤ R rounds)
```

**Stage responsibilities**

1. **Profile** — Column dtypes, cardinality, missingness, distributions, target association sketch, pairwise correlation of numerics. If an `eda_agent` profile artifact exists for the dataset, ingest it instead of recomputing.
2. **Generate** — LLM proposes N candidate feature *specs* (not free code) from the profile + user-supplied domain context. Structured output, schema-validated, deduplicated against prior rounds.
3. **Validate + Materialize** — Static validation (allowed columns, no target references, allowed ops), then execution through the safe executor. Invalid candidates are rejected with a machine-readable reason that feeds back to the generator.
4. **Evaluate** — Stratified/grouped K-fold CV. Baseline model (original features) vs augmented model (original + candidates). Per-candidate attribution via out-of-fold permutation importance. Shadow noise features injected as a significance benchmark.
5. **Select + Adjudicate** — Deterministic gates first (shadow gate, redundancy clustering, stability check). The LLM then reviews an *evidence pack* for borderline candidates only — it can flag a statistically weak feature for targeted segment evaluation, but cannot keep a feature without evidence.
6. **Report** — Feature registry (JSON), serialized sklearn `Pipeline` reproducing the kept features, and a human-readable `report.md` with per-feature rationale, written by the LLM from the evidence pack.

---

## 4. Data Contracts

All LLM I/O and inter-stage handoffs are Pydantic models. The generator emits `FeatureSpec`, never free-form code.

```python
from pydantic import BaseModel, Field
from typing import Literal

class FeatureSpec(BaseModel):
    """One candidate feature. Either a declarative op (preferred) or a
    row-wise expression validated by the AST whitelist."""
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,40}$")
    kind: Literal["op", "expression"]
    # kind == "op": a named transform with parameters
    op: str | None = None            # e.g. "ratio", "log1p", "date_diff_days",
                                     # "group_stat", "count_encode", "bin_quantile"
    inputs: list[str] = []           # source columns
    params: dict = {}                # op-specific params, schema-checked per op
    # kind == "expression": row-wise stateless pandas/numpy expression
    expression: str | None = None    # e.g. "monthly_spend / (days_since_login + 1)"
    rationale: str                   # one-sentence domain hypothesis
    hypothesis_segment: str | None = None  # optional: "matters most for enterprise tier"

class CandidateResult(BaseModel):
    spec: FeatureSpec
    status: Literal["kept", "pruned", "rejected_invalid", "rejected_leaky", "flagged"]
    oof_permutation_importance: float | None = None
    shadow_percentile: float | None = None   # importance vs. shadow distribution
    redundancy_cluster: int | None = None
    max_abs_corr_with_existing: float | None = None
    fold_stability: float | None = None      # sign-consistency of importance across folds
    decision_rationale: str

class RoundSummary(BaseModel):
    round_index: int
    baseline_metric: float           # mean CV metric, original features
    augmented_metric: float          # mean CV metric, original + kept candidates
    metric_std: float
    candidates: list[CandidateResult]
```

**Why specs beat formula strings:** every `op` has a known fold-safety class (see §6.3), a parameter schema to validate, and a deterministic implementation we control. The `expression` escape hatch stays row-wise-only, which makes it leakage-safe by construction, and is AST-validated (§6.2).

---

## 5. Repository Layout

Mirrors the standalone-agent convention established by `eda_agent`.

```
feature_agent/
├── __init__.py
├── config.py            # FeatureAgentConfig dataclass; YAML-loadable
├── llm.py               # OpenRouter client wrapper: retries, JSON-schema
│                        #   enforcement, token/cost accounting
├── profile.py           # dataset profiling + eda_agent artifact ingestion
├── prompts/
│   ├── generate.md      # candidate generation prompt template
│   ├── adjudicate.md    # borderline-review prompt template
│   └── report.md        # final report prompt template
├── generate.py          # candidate generation + dedup across rounds
├── ops.py               # declarative op library (ratio, log1p, group_stat, ...)
│                        #   each op = sklearn-compatible transformer
├── executor.py          # spec → column materialization; AST whitelist for
│                        #   expression-kind specs
├── guards.py            # leakage validator, name/column checks, canaries
├── evaluate.py          # CV harness: baseline vs augmented, OOF permutation
│                        #   importance, shadow features
├── select.py            # shadow gate, redundancy clustering, stability,
│                        #   LLM adjudication of borderline cases
├── report.py            # registry writer, pipeline serialization, report.md
├── orchestrator.py      # round loop, budgets, manifest, public API
└── tests/               # see §10
```

Public API kept to one call:

```python
from feature_agent import FeatureAgent, FeatureAgentConfig

result = FeatureAgent(FeatureAgentConfig(model="<openrouter-model-id>")).run(
    df, target="churned", task="classification",
    domain_context="B2B SaaS subscription business; churn = no renewal within 30 days",
)
result.kept_features      # list[CandidateResult]
result.pipeline           # sklearn Pipeline: raw df -> engineered matrix
result.report_path        # report.md
```

---

## 6. Key Design Decisions

### 6.1 LLM access — OpenRouter only

Single client wrapper in `llm.py`; no other module touches the network. OpenRouter is OpenAI-API-compatible, so the `openai` SDK is reused with a different base URL:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

Wrapper responsibilities (all three call sites go through it):

- **Model from config**, never hardcoded. Config carries a `model` (primary) and `fallback_model` (cheaper, used for the report stage where reasoning demand is lower).
- **Structured output enforcement**: request JSON, parse into the Pydantic model, and on validation failure retry up to 2× with the validation error appended to the prompt. Hard-fail the candidate batch (not the run) after retries.
- **Determinism posture**: `temperature=0.4` for generation (diversity is useful), `0.0` for adjudication and reporting. Prompt + response hashes recorded in the run manifest.
- **Budget accounting**: token counts and cost per call accumulated; orchestrator enforces a per-run cost ceiling (§8).

### 6.2 Safe execution — no raw code from the LLM

Two tiers, both closed-world:

**Tier 1 — declarative ops (preferred, covers ~90% of useful features).** The op library in `ops.py` defines a fixed vocabulary, each implemented as an sklearn-compatible transformer: `ratio`, `product`, `difference`, `log1p`, `sqrt`, `clip`, `bin_quantile`, `date_diff_days`, `date_part`, `count_encode`, `freq_encode`, `group_stat` (mean/median/std of a numeric within a categorical group), `target_encode` (v1.1, fold-fitted only), `is_missing`, `interaction_flag`. The generation prompt lists this vocabulary with parameter schemas; the LLM composes from it.

**Tier 2 — row-wise expressions (escape hatch).** For arithmetic the op vocabulary doesn't express cleanly. Validated by AST inspection before execution:

```python
import ast

ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
                 ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                 ast.USub, ast.Call, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE)
ALLOWED_FUNCS = {"log1p", "sqrt", "abs", "clip", "where"}  # resolved to numpy only

def validate_expression(expr: str, allowed_columns: set[str]) -> None:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise InvalidCandidate(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_columns | ALLOWED_FUNCS:
            raise InvalidCandidate(f"unknown name: {node.id}")
        if isinstance(node, ast.Call) and node.func.id not in ALLOWED_FUNCS:
            raise InvalidCandidate(f"disallowed function: {node.func.id}")
```

No attribute access, no subscripts, no imports, no dunders — those node types simply aren't in the whitelist. Execution happens against a namespace containing only the dataframe's columns as Series and the whitelisted numpy functions. `allowed_columns` **excludes the target** and any user-declared leaky columns, so a formula referencing them fails validation rather than needing post-hoc detection.

Rejected candidates return a structured reason (`rejected_invalid` + message) that is included in the next generation round's prompt, so the LLM learns the boundary instead of repeating mistakes.

### 6.3 Leakage defense — three layers

Leakage is the highest-severity failure mode for this agent: a leaky feature looks *spectacular* in evaluation and poisons the downstream model. Defense in depth:

1. **Static (pre-execution).** The leakage validator in `guards.py` blocks: any reference to the target; user-declared post-outcome columns (config: `forbidden_columns`, e.g. `cancellation_date` in a churn task); and `group_stat`/`target_encode` ops grouped on near-unique keys (cardinality > 50% of rows — effectively row identity).
2. **Structural (during evaluation).** Fitted transforms (`group_stat`, `count_encode`, `bin_quantile`, `target_encode`) are sklearn transformers **fitted inside each CV fold on training data only** — never on the full frame. Row-wise ops and expressions are stateless and safe by construction. This is the reason ops are transformers rather than functions.
3. **Empirical (post-evaluation).** Suspicion heuristics flag rather than silently keep: single-feature AUC > 0.95 (classification) or |Spearman| > 0.95 with target (regression); a candidate whose addition improves CV metric by an implausible jump (> 10× the std of baseline fold scores). Flagged features are excluded from `kept` and surfaced prominently in the report — a human confirms whether it's leakage or a genuine gold feature.

### 6.4 Evaluation protocol — marginal lift, not standalone importance

The question is never "is this feature important among the candidates?" — it's "does this feature add signal **beyond what the baseline already has**?" Protocol per round:

1. **Preprocessing (typed, not `fillna(0)`).** Numerics: leave NaN (LightGBM handles natively) plus optional `is_missing` indicators. Categoricals: cast to `category` dtype for native handling. Dates: excluded raw; only reachable via date ops.
2. **Baseline CV.** LightGBM (`LGBMClassifier`/`LGBMRegressor` per task) on original features. Stratified K-fold (classification) or K-fold (regression); `GroupKFold` when the user declares an entity column (repeated customers must not straddle folds). Metric: AUC / average-precision for classification (config), RMSE / MAE for regression. Record mean ± std across folds.
3. **Augmented CV.** Same folds (identical seed/splits — paired comparison), original + all valid candidates + **S shadow features** (S = max(5, ⌈0.5 × n_candidates⌉): random permutations of real candidate columns, preserving marginal distributions).
4. **Attribution.** Permutation importance computed on **out-of-fold data per fold**, averaged across folds. Chosen over split-gain (biased toward high-cardinality features) and over SHAP for gating (SHAP is retained for the report's direction/segment narratives, computed once on the final kept set).
5. **Shadow gate.** A candidate survives only if its mean OOF permutation importance exceeds the **95th percentile of the shadow features' importances**. This replaces the prototype's fixed 0.01 threshold with a data-adaptive null distribution — the Boruta idea without Boruta's runtime.
6. **Redundancy pruning.** Spearman correlation matrix over survivors **plus original features**; hierarchical clustering at |ρ| > 0.9; keep the highest-importance member per cluster. A candidate that merely re-expresses an original feature dies here.
7. **Stability check.** Sign/rank consistency of importance across folds; features whose importance is driven by a single fold are demoted to borderline.
8. **Confirmation ablation.** Final CV run: baseline + kept-only. Required: confirmation metric ≥ baseline + `min_lift` (config, default: 0.5 × baseline fold std). If the kept set fails confirmation, the round reports zero keeps rather than shipping noise.

### 6.5 LLM adjudication — reasoning constrained by evidence

The prototype's instinct — "the agent reasons about scores before pruning; a feature can look weak globally but carry segment signal" — is right, but unconstrained LLM override of statistics is how noise gets shipped. Constrained version:

- Deterministic gates (§6.4) produce three sets: **clear keeps**, **clear prunes**, **borderline** (shadow percentile 75–95, or unstable, or spec carried a `hypothesis_segment`).
- The LLM sees an evidence pack for borderline candidates only: importance stats, shadow percentile, fold-level scores, correlation neighbors, and — when a `hypothesis_segment` exists — the metric delta computed *within that segment*.
- Allowed verdicts per candidate: `prune`, or `request_segment_eval` naming a concrete segment expression. The orchestrator then actually runs the segment-restricted evaluation; the feature is kept only if it passes the shadow gate **within the segment**. The LLM can never emit `keep` directly — evidence does.

### 6.6 Iteration loop

Rounds ≤ R (default 3). Each round's generation prompt includes: survivors with their importance ranks, prunes with reasons, invalid/leaky rejections with validator messages, and the current lift over baseline. Stopping conditions (any): round lift < `min_round_lift`; two consecutive rounds with zero keeps; cost or wall-time budget reached. This mirrors how a human iterates — double down on the shapes of features that worked (ratios survived, raw interactions didn't → propose more ratios).

---

## 7. Outputs & Reproducibility

Every run writes a self-contained artifact directory:

```
runs/<run_id>/
├── manifest.json          # dataset hash, config, model id, prompt hashes,
│                          #   seeds, package versions, cost, wall time
├── feature_registry.json  # every candidate ever proposed: spec, status,
│                          #   scores, decision rationale, round index
├── pipeline.joblib        # sklearn Pipeline: raw df -> engineered matrix
│                          #   (kept features only, fold-fitted transforms
│                          #   refit on full training data at export time)
├── report.md              # LLM-written narrative: what was tried, what
│                          #   survived and why, flagged suspicions,
│                          #   suggested next experiments
└── rounds/round_<k>.json  # RoundSummary per round
```

The registry records *pruned and rejected* candidates too — "we tried `tickets_per_spend_ratio` and it failed the shadow gate" is exactly the documentation the manual process never produces, and it prevents the next person (or next run) from re-proposing dead ends.

---

## 8. Configuration

```python
@dataclass
class FeatureAgentConfig:
    # LLM (OpenRouter)
    model: str                       # primary model id, e.g. from team default
    fallback_model: str | None = None  # cheaper model for report stage
    generation_temperature: float = 0.4
    max_cost_usd: float = 2.00       # hard ceiling per run

    # Generation
    n_candidates_per_round: int = 15
    max_rounds: int = 3

    # Evaluation
    n_folds: int = 5
    cv_metric: str = "auto"          # auc | average_precision | rmse | mae
    group_column: str | None = None  # entity id for GroupKFold
    forbidden_columns: list[str] = field(default_factory=list)
    shadow_percentile: float = 95.0
    redundancy_rho: float = 0.90
    min_lift: str | float = "0.5*std"  # confirmation-ablation requirement
    random_state: int = 42

    # Sampling for large data
    max_eval_rows: int = 200_000     # stratified subsample above this
```

---

## 9. Implementation Plan

Six milestones, each independently testable with a demoable acceptance criterion. Estimated total: **~2.5 engineer-weeks.**

| Milestone | Scope | Acceptance criterion | Est. |
|---|---|---|---|
| **M0 — Scaffolding** | Package skeleton, `config.py`, `llm.py` (OpenRouter wrapper: retries, Pydantic-validated JSON, cost accounting), run-manifest writer | Wrapper returns a validated Pydantic object from a live OpenRouter call; malformed-JSON path exercised via mock; cost ceiling aborts cleanly | 1 d |
| **M1 — Profile + Generate** | `profile.py` (incl. `eda_agent` artifact ingestion), prompt templates, `generate.py` with dedup | On the synthetic churn dataset, produces ≥ 12/15 schema-valid, non-duplicate specs referencing only real columns | 2 d |
| **M2 — Executor + Guards** | `ops.py` transformer library, `executor.py` AST validation, `guards.py` static leakage checks | Adversarial suite passes: expressions with imports/dunders/subscripts/target-refs all rejected with structured reasons; every op round-trips fit/transform on fold splits | 3 d |
| **M3 — Evaluation harness** | `evaluate.py`: paired baseline/augmented CV, OOF permutation importance, shadow features, typed preprocessing, GroupKFold support | On a synthetic dataset with one planted signal feature and five noise candidates, the planted feature beats the shadow gate and all five noise candidates fail it, across 10 seeds | 3 d |
| **M4 — Selection + Reporting** | `select.py` (gates, redundancy clustering, stability, LLM adjudication), `report.py` (registry, pipeline export, report.md) | End-to-end single-round run on the churn scenario produces all artifacts in §7; exported pipeline reproduces the engineered matrix bit-exactly on a fresh dataframe | 2 d |
| **M5 — Iteration + Hardening** | `orchestrator.py` round loop with stopping rules and budgets; full test suite (§10); README + usage docs | 3-round run stays under cost ceiling; benchmark suite (§11) meets targets; leakage canary test red-teams the full pipeline | 3 d |

Dependency order is strict M0 → M1 → M2 → M3 → M4 → M5, but M2 and M3 can proceed in parallel after M1 if a second pair of hands is available (executor and evaluator share only the `FeatureSpec` contract).

---

## 10. Testing Strategy

- **Unit, no network.** All LLM interactions mocked with recorded fixtures. Executor, guards, evaluation math, and gates are pure functions of data — property-test them (e.g., Hypothesis-generated expressions must never execute if validation rejected them).
- **Adversarial executor suite.** Expressions attempting `__import__`, attribute access, subscripting, `os`/`open` calls, references to the target, and names outside the schema — every one must raise `InvalidCandidate`, never execute.
- **Leakage canaries (the critical test).** Two planted traps run against the *full* pipeline: (a) a copy of the target under an innocent name in `forbidden_columns` — must be statically rejected; (b) a 98%-correlated proxy *not* declared forbidden — must end `flagged`, never `kept`.
- **Recovery benchmark.** Synthetic datasets with planted ground truth (e.g., target depends on `x1/x2` and `log(x3)`): assert the agent recovers the planted forms within 2 rounds at ≥ 80% rate over 10 seeds, with false-keep rate ≤ 10%.
- **Determinism.** Same data + config + recorded LLM fixtures → identical registry and pipeline.
- **Golden E2E.** One live-LLM smoke test (marked, excluded from CI default) on the churn scenario, asserting artifact completeness rather than exact content.

---

## 11. Success Metrics

For the agent itself, measured on a fixed benchmark suite (3 synthetic + 2 public datasets, e.g. Telco Churn and House Prices):

- **Lift:** mean CV metric improvement of baseline+kept over baseline. Target: positive on ≥ 4/5 benchmark datasets, with confirmation ablation passing.
- **Precision of selection:** false-keep rate on synthetic benchmarks ≤ 10%; planted-feature recovery ≥ 80%.
- **Candidate validity:** ≥ 80% of generated specs pass validation and execute (measures prompt/vocabulary quality).
- **Safety:** zero leakage-canary escapes, ever — this is a release gate, not a metric to trend.
- **Cost & latency:** ≤ $2 and ≤ 15 min per 3-round run on a 200k-row dataset.

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM proposes subtle leaky features (proxy columns) | Medium | Severe | Three-layer defense (§6.3); flagged-not-kept default; canary release gate |
| Permutation importance unstable on small datasets | Medium | Medium | Fold-stability check; shadow gate is relative, not absolute; warn below 1k rows |
| Correlated candidates split importance and both die | Medium | Medium | Redundancy clustering runs *before* final gating verdicts; cluster importance evaluated jointly |
| Op vocabulary too narrow → LLM fights the schema | Medium | Low | Rejection reasons fed back into prompts; vocabulary extension is a one-file change in `ops.py`; expression tier as pressure valve |
| Cost blowup from iteration loop | Low | Low | Hard cost ceiling, round cap, plateau stopping rule |
| OpenRouter model deprecation / rate limits | Low | Medium | Model id in config; wrapper retries with exponential backoff; fallback model |

---

## 13. Extensions (explicitly out of v1)

1. **Time-series ops** — lags, rolling windows, expanding stats with time-aware CV (`TimeSeriesSplit`). Largest user demand; needs its own leakage discipline (temporal, not just fold).
2. **Target encoding tier** — fold-fitted `target_encode` op is implemented in M2 but disabled by default; enable after the canary suite covers it specifically.
3. **Multi-table synthesis** — join-aware `group_stat` across relations.
4. **Cross-agent orchestration** — `eda_agent` profile → this agent → a future model-selection agent, sharing the run-manifest convention.

---

## 14. Reference Scenario (acceptance narrative)

Customer churn prediction, 12 input columns including `days_since_login`, `plan_tier`, `support_tickets_90d`, `monthly_spend`. Config declares `forbidden_columns=["cancellation_date"]` and `group_column="customer_id"`.

Round 1: the agent proposes 15 candidates. Two are rejected statically (one referenced `cancellation_date`, one used a disallowed function). Twelve of the remaining thirteen execute; baseline AUC is 0.741 ± 0.008, augmented 0.769. Seven candidates beat the shadow gate; redundancy clustering merges `spend_per_day` and `daily_spend_rate` (ρ = 0.97), keeping the former. `login_recency_x_plan` lands borderline (shadow percentile 88) with `hypothesis_segment="enterprise tier"`; the adjudicator requests a segment evaluation, where it clears the gate within enterprise rows and is kept with that caveat recorded. Round 2 proposes ratio-heavy variants (the feedback said ratios survived), two more keep. Round 3 yields zero keeps → stop.

Final: 8 features kept, confirmation ablation AUC 0.768 ± 0.007 (baseline + 0.027, > min_lift). The report calls out `tickets_per_spend_ratio` as the top feature (OOF permutation importance 3.1× the shadow ceiling) with the SHAP-derived narrative: *"high-spend customers who also raise support tickets are a distinctly elevated churn risk"* — a finding routed to the product team. The registry documents all 30 candidates tried across rounds, including the 13 that died and why.
