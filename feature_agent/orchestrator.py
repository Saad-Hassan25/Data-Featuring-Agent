"""The deterministic controller: round loop, budgets, manifest, public API (§3, §6.6).

Six pipeline stages run per round — profile (once), generate, validate+materialize,
evaluate, select+adjudicate, and finally report. The LLM is called at three points
(generation, adjudication, reporting); everything else is deterministic so runs are
reproducible and cheap. Rounds iterate until lift plateaus, two rounds keep nothing,
or a budget (cost / wall-time / round cap) is hit.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FeatureAgentConfig
from .evaluate import Evaluator
from .executor import dry_run
from .generate import generate_candidates
from .guards import InvalidCandidate, LeakageError, allowed_columns, validate_spec_static
from .llm import BudgetExceeded, LLMClient
from .profile import build_profile
from .report import (
    ReportContext,
    deterministic_report,
    export_pipeline,
    llm_report,
    render_markdown,
    write_registry,
    write_round,
)
from .schemas import (
    CandidateResult,
    DataProfile,
    FeatureReportNarrative,
    FeatureSpec,
    RoundSummary,
    TaskType,
)
from .select import select_round


@dataclass
class FeatureAgentResult:
    kept_features: list[CandidateResult]
    pipeline: object                       # sklearn Pipeline: raw df -> engineered matrix
    report_path: str
    run_dir: str
    registry: list[CandidateResult] = field(default_factory=list)
    rounds: list[RoundSummary] = field(default_factory=list)
    narrative: FeatureReportNarrative | None = None
    baseline_metric: float = 0.0
    final_metric: float = 0.0
    lift: float = 0.0
    confirmation_passed: bool = False
    metric_name: str = ""
    cost_usd: float = 0.0
    manifest: dict = field(default_factory=dict)


class FeatureAgent:
    """Public API — one call. See §5."""

    def __init__(self, config: FeatureAgentConfig | None = None):
        self.config = config or FeatureAgentConfig.from_env()

    # ------------------------------------------------------------------ #
    def run(
        self,
        df: pd.DataFrame,
        target: str,
        task: TaskType,
        domain_context: str = "",
        eda_report: str | Path | None = None,
    ) -> FeatureAgentResult:
        cfg = self.config
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        np.random.seed(cfg.random_state)
        self._run_id_cache = None  # fresh run id per .run() call
        t0 = time.time()

        profile = build_profile(df, target, task, cfg, eda_report)
        evaluator = Evaluator(df, profile, cfg)
        llm = LLMClient(cfg)
        llm_ok, why = llm.available()
        if cfg.use_llm and not llm_ok:
            self._log(f"LLM unavailable ({why}) — using the deterministic generator/report.")
        active_llm = llm if llm_ok else None

        allowed = allowed_columns(profile, cfg)
        y_series = pd.Series(evaluator.y, index=evaluator.df.index)

        registry: list[CandidateResult] = []
        rounds: list[RoundSummary] = []
        history: list[RoundSummary] = []
        kept_specs: list[FeatureSpec] = []
        seen_signatures: set[str] = set()
        used_names: set[str] = set()
        zero_keep_streak = 0
        prev_cum_lift = 0.0

        for r in range(cfg.max_rounds):
            if self._budget_hit(llm, t0):
                self._log("Budget reached — stopping before a new round.")
                break

            candidates = generate_candidates(profile, cfg, domain_context,
                                              active_llm, history, seen_signatures)
            if not candidates:
                self._log(f"Round {r + 1}: generator produced no new candidates — stopping.")
                break
            for s in candidates:
                seen_signatures.add(s.signature())

            valid, round_results = self._validate(candidates, profile, cfg, evaluator,
                                                   allowed, y_series, used_names, r + 1)
            self._log(f"Round {r + 1}: {len(candidates)} proposed, {len(valid)} valid.")

            if valid:
                ev = evaluator.evaluate_candidates(valid, carried=kept_specs)
                results, new_kept, confirmation = select_round(
                    valid, ev, evaluator, profile, cfg, active_llm, r + 1, carried_specs=kept_specs)
            else:
                ev = evaluator.evaluate_candidates([], carried=kept_specs)
                results, new_kept = [], []
                confirmation = evaluator.confirmation(kept_specs)

            round_results.extend(results)
            registry.extend(round_results)
            kept_specs.extend(new_kept)
            for c in round_results:
                used_names.add(c.spec.name)

            summary = self._round_summary(r + 1, ev, round_results, len(new_kept),
                                          confirmation, len(candidates))
            rounds.append(summary)
            history.append(summary)
            write_round(summary, self._rounds_dir(profile, t0) / f"round_{r + 1}.json")
            self._log(f"Round {r + 1}: {len(new_kept)} kept "
                      f"(cumulative lift {confirmation.lift:+.4f}, "
                      f"{'confirmed' if confirmation.passed else 'not confirmed'}).")

            zero_keep_streak = zero_keep_streak + 1 if not new_kept else 0
            incremental = confirmation.lift - prev_cum_lift
            prev_cum_lift = confirmation.lift
            if zero_keep_streak >= 2:
                self._log("Two consecutive rounds with zero keeps — stopping.")
                break
            if r > 0 and incremental < cfg.min_round_lift:
                self._log(f"Round lift {incremental:+.4f} < min_round_lift "
                          f"{cfg.min_round_lift} — stopping.")
                break

        return self._finalize(df, profile, evaluator, cfg, active_llm, llm, registry,
                              rounds, kept_specs, y_series, t0)

    # ------------------------------------------------------------------ #
    def _validate(self, candidates, profile, cfg, evaluator, allowed, y_series,
                  used_names, round_index) -> tuple[list[FeatureSpec], list[CandidateResult]]:
        valid: list[FeatureSpec] = []
        rejected: list[CandidateResult] = []
        local_names = set(used_names)
        for spec in candidates:
            try:
                validate_spec_static(spec, profile, cfg, existing_names=local_names)
                dry_run(spec, evaluator.df, allowed, y=y_series)
            except LeakageError as exc:
                rejected.append(CandidateResult(
                    spec=spec, status="rejected_leaky", round_index=round_index,
                    decision_rationale=f"static leakage guard: {exc}"))
                continue
            except InvalidCandidate as exc:
                rejected.append(CandidateResult(
                    spec=spec, status="rejected_invalid", round_index=round_index,
                    decision_rationale=str(exc)))
                continue
            valid.append(spec)
            local_names.add(spec.name)
        return valid, rejected

    def _round_summary(self, idx, ev, results, kept_count, confirmation, n_proposed) -> RoundSummary:
        return RoundSummary(
            round_index=idx, metric_name=ev.metric_name,
            baseline_metric=round(ev.baseline_mean, 6), augmented_metric=round(ev.augmented_mean, 6),
            metric_std=round(ev.baseline_std, 6),
            baseline_fold_scores=[round(s, 6) for s in ev.baseline_scores],
            augmented_fold_scores=[round(s, 6) for s in ev.augmented_scores],
            n_shadows=ev.n_shadows, n_candidates=n_proposed, kept_count=kept_count,
            confirmation_metric=round(confirmation.ablation_mean, 6),
            confirmation_passed=confirmation.passed,
            lift_over_baseline=round(confirmation.lift, 6),
            candidates=results,
        )

    def _finalize(self, df, profile, evaluator, cfg, active_llm, llm, registry,
                  rounds, kept_specs, y_series, t0) -> FeatureAgentResult:
        run_dir = self._run_dir(profile, t0)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "rounds").mkdir(exist_ok=True)

        kept_results = [c for c in registry if c.status == "kept"]
        final_kept_specs = [c.spec for c in kept_results]
        conf = evaluator.confirmation(final_kept_specs)

        pipeline = export_pipeline(final_kept_specs, evaluator.allowed, evaluator.df, y_series)
        joblib.dump(pipeline, run_dir / "pipeline.joblib")
        write_registry(registry, run_dir / "feature_registry.json")

        flagged = [c for c in registry if c.status == "flagged"]
        notable = self._notable_prunes(registry)
        ctx = ReportContext(
            profile=profile, metric_name=conf.metric_name,
            baseline_metric=conf.baseline_mean, final_metric=conf.ablation_mean,
            lift=conf.lift, min_lift=conf.min_lift,
            confirmation_passed=bool(final_kept_specs) and conf.passed,
            kept=kept_results, flagged=flagged, notable_prunes=notable, rounds=rounds,
        )
        narrative, source = llm_report(active_llm, ctx, cfg)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        md = render_markdown(narrative, ctx, source, generated_at)
        report_path = run_dir / "report.md"
        report_path.write_text(md, encoding="utf-8")

        manifest = self._manifest(df, profile, cfg, llm, rounds, kept_results, conf, t0, source)
        (run_dir / "manifest.json").write_text(
            _json(manifest), encoding="utf-8")

        self._log(f"Done. {len(kept_results)} features kept | lift {conf.lift:+.4f} | "
                  f"cost ${llm.budget.cost_usd:.4f} | {run_dir}")
        return FeatureAgentResult(
            kept_features=kept_results, pipeline=pipeline, report_path=str(report_path),
            run_dir=str(run_dir), registry=registry, rounds=rounds, narrative=narrative,
            baseline_metric=conf.baseline_mean, final_metric=conf.ablation_mean,
            lift=conf.lift, confirmation_passed=ctx.confirmation_passed,
            metric_name=conf.metric_name, cost_usd=llm.budget.cost_usd, manifest=manifest,
        )

    # ------------------------------------------------------------------ #
    def _notable_prunes(self, registry) -> list[CandidateResult]:
        prunes = [c for c in registry if c.status in ("pruned", "rejected_invalid", "rejected_leaky")]
        prunes.sort(key=lambda c: (c.oof_permutation_importance or 0.0), reverse=True)
        return prunes[:15]

    def _manifest(self, df, profile, cfg, llm, rounds, kept, conf, t0, source) -> dict:
        return {
            "run_id": self._compute_run_id(profile, t0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"hash": self._dataset_hash(df), "n_rows": int(len(df)),
                        "n_cols": int(df.shape[1]), "target": profile.target,
                        "task": profile.task, "profile_source": profile.source},
            "config": cfg.to_dict(),
            "model": {"primary": cfg.model, "fallback": cfg.fallback_model,
                      "report_source": source},
            "seeds": {"random_state": cfg.random_state},
            "result": {"metric": conf.metric_name, "baseline": round(conf.baseline_mean, 6),
                       "final": round(conf.ablation_mean, 6), "lift": round(conf.lift, 6),
                       "confirmation_passed": bool(kept) and conf.passed,
                       "n_kept": len(kept), "n_rounds": len(rounds)},
            "llm_calls": [c.__dict__ for c in llm.call_log],
            "cost_usd": round(llm.budget.cost_usd, 6),
            "tokens": {"prompt": llm.budget.prompt_tokens, "completion": llm.budget.completion_tokens},
            "wall_seconds": round(time.time() - t0, 2),
            "package_versions": _versions(),
        }

    # -- budgets / ids / logging ------------------------------------------ #
    def _budget_hit(self, llm: LLMClient, t0: float) -> bool:
        if llm.budget.would_exceed():
            return True
        if self.config.max_wall_seconds and (time.time() - t0) > self.config.max_wall_seconds:
            return True
        return False

    _run_id_cache: str | None = None

    def _compute_run_id(self, profile: DataProfile, t0: float) -> str:
        if self._run_id_cache is None:
            self._run_id_cache = self.config.run_id or f"{profile.task}-{int(t0)}"
        return self._run_id_cache

    def _run_dir(self, profile: DataProfile, t0: float) -> Path:
        return Path(self.config.output_dir) / self._compute_run_id(profile, t0)

    def _rounds_dir(self, profile: DataProfile, t0: float) -> Path:
        d = self._run_dir(profile, t0) / "rounds"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _dataset_hash(df: pd.DataFrame) -> str:
        h = hashlib.sha256()
        h.update(str(df.shape).encode())
        h.update("|".join(map(str, df.columns)).encode())
        try:
            h.update(pd.util.hash_pandas_object(df.head(1000), index=True).to_numpy().tobytes())
        except Exception:
            pass
        return h.hexdigest()[:16]

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[feature-agent] {msg}", file=sys.stderr)


def _versions() -> dict:
    import importlib
    out = {}
    for pkg in ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "pydantic"):
        mod = pkg.replace("scikit-learn", "sklearn")
        try:
            out[pkg] = importlib.import_module(mod).__version__
        except Exception:
            out[pkg] = "not installed"
    return out


def _json(obj: dict) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
