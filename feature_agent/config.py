"""Central configuration for the Feature Engineering & Selection agent.

Design principle (shared with the EDA agent): every knob that changes what the
agent *does* — the statistical gates, the CV protocol, the LLM connection, the
budgets — lives here, documented and tunable, instead of being scattered as magic
numbers. A principal data scientist should be able to read this file and know
exactly how candidates are proposed, judged, and shipped.

The config is a plain dataclass so it is trivially constructable in code, loadable
from YAML, and serializable into the run manifest.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Minimal .env loader (no hard dependency on python-dotenv)
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class FeatureAgentConfig:
    """All settings for one run of the feature agent.

    Fields mirror the design doc (§8) and add the operational plumbing needed to
    actually run: the OpenRouter connection, budgets, and the stopping rules for
    the iteration loop.
    """

    # --- LLM (OpenRouter, OpenAI-compatible) --------------------------------- #
    model: str = "anthropic/claude-sonnet-4"      # primary model id
    fallback_model: str | None = None             # cheaper model for the report stage
    generation_temperature: float = 0.4           # diversity helps generation
    adjudication_temperature: float = 0.0         # determinism for judgment/reporting
    max_cost_usd: float = 2.00                    # hard per-run cost ceiling
    # Price estimates (USD per 1M tokens) used for the cost ceiling when the API
    # does not report an exact cost. Override per model. Defaults are a rough
    # sonnet-class estimate — the ceiling stays conservative, never silent.
    input_price_per_mtok: float = 3.0
    output_price_per_mtok: float = 15.0
    max_output_tokens: int = 4096
    request_timeout: float = 120.0
    # OpenRouter connection / attribution (usually read from env, see from_env).
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = ""
    app_name: str = "feature-agent"
    use_llm: bool = True                          # False -> deterministic generator/report

    # --- Generation ---------------------------------------------------------- #
    n_candidates_per_round: int = 15
    max_rounds: int = 3
    enable_target_encode: bool = False            # fold-fitted target encoding (§13); off by default

    # --- Evaluation ---------------------------------------------------------- #
    n_folds: int = 5
    cv_metric: str = "auto"                       # auc | average_precision | rmse | mae | auto
    group_column: str | None = None              # entity id -> GroupKFold
    forbidden_columns: list[str] = field(default_factory=list)  # post-outcome / leaky cols
    id_columns: list[str] = field(default_factory=list)         # extra cols to exclude from features
    shadow_percentile: float = 95.0               # candidate must beat this shadow percentile
    borderline_percentile: float = 75.0           # [borderline, shadow] band -> LLM adjudication
    redundancy_rho: float = 0.90                  # |Spearman| cluster threshold
    stability_min: float = 0.50                   # min sign-consistency across folds to be "stable"
    min_lift: str | float = "0.5*std"             # confirmation-ablation requirement
    min_shadow_features: int = 5                  # S = max(this, ceil(shadow_fraction * n_candidates))
    shadow_fraction: float = 0.5
    random_state: int = 42

    # --- Empirical leakage heuristics (§6.3, layer 3) ------------------------ #
    leakage_single_auc: float = 0.95              # single-feature AUC above -> flag
    leakage_single_corr: float = 0.95             # |Spearman| with target above -> flag
    implausible_jump_std_mult: float = 10.0       # metric jump > this * baseline std -> flag

    # --- Iteration stopping rules (§6.6) ------------------------------------- #
    min_round_lift: float = 0.0                   # stop if a round's lift over baseline <= this
    max_wall_seconds: float | None = 900.0        # 15-minute wall budget (None disables)

    # --- Sampling for large data --------------------------------------------- #
    max_eval_rows: int = 200_000                  # stratified subsample above this

    # --- Output -------------------------------------------------------------- #
    output_dir: str = "feature_runs"              # runs/<run_id> is created underneath
    run_id: str | None = None                     # explicit id (else derived from data hash + time)
    make_shap: bool = True                        # SHAP narratives for the report (best-effort)
    verbose: bool = True

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2 for cross-validation.")
        if not (0 < self.shadow_percentile <= 100):
            raise ValueError("shadow_percentile must be in (0, 100].")
        if not (0 <= self.borderline_percentile <= self.shadow_percentile):
            raise ValueError("borderline_percentile must be in [0, shadow_percentile].")

    # ------------------------------------------------------------------ #
    def n_shadows(self, n_candidates: int) -> int:
        """S = max(min_shadow_features, ceil(shadow_fraction * n_candidates))."""
        return max(self.min_shadow_features, math.ceil(self.shadow_fraction * max(n_candidates, 0)))

    def resolve_min_lift(self, baseline_fold_std: float) -> float:
        """Resolve the confirmation-ablation requirement to an absolute number.

        Accepts a float (absolute lift) or a string of the form "<factor>*std"
        (a multiple of the baseline's across-fold std).
        """
        v = self.min_lift
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower().replace(" ", "")
        if s.endswith("*std"):
            factor = float(s[:-4] or "1")
            return factor * float(baseline_fold_std)
        return float(s)  # bare numeric string

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **overrides: Any) -> "FeatureAgentConfig":
        """Build a config from environment / .env, then apply keyword overrides.

        Precedence: explicit overrides > environment > dataclass defaults.
        `None` overrides are ignored so CLI flags can be passed through blindly.
        """
        load_dotenv()
        cfg = cls(
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            fallback_model=os.getenv("OPENROUTER_FALLBACK_MODEL") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            site_url=os.getenv("OPENROUTER_SITE_URL", ""),
            app_name=os.getenv("OPENROUTER_APP_NAME", "feature-agent"),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown setting: {key}")
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str], **overrides: Any) -> "FeatureAgentConfig":
        """Load a config from a YAML file (requires PyYAML), env for secrets."""
        try:
            import yaml  # optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError("from_yaml needs PyYAML: pip install pyyaml") from exc
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_env(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serializable view for the run manifest (never emits the API key)."""
        d = asdict(self)
        d.pop("openrouter_api_key", None)
        return d
