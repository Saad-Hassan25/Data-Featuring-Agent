"""Dataset profiling + eda_agent artifact ingestion (§3 stage 1).

The generator reasons over a compact profile: column semantic types, cardinality,
missingness, a numeric sketch, and a cheap target-association hint. If an
`eda_agent` report (`report.json`) exists for the dataset we ingest its richer
profile instead of recomputing — the read-only companion dependency from §1. If
it is absent or unreadable we fall back to internal profiling, so the agent never
hard-depends on the EDA agent having run.

None of these numbers gate a feature; gating uses the CV harness (§6.4). The
profile exists to ground the LLM's proposals in the data's shape.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FeatureAgentConfig
from .schemas import ColumnProfile, DataProfile, SemanticType, TaskType

_BOOL_STRINGS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1", "on", "off"}


def _is_texty(s: pd.Series) -> bool:
    """object OR pandas-3 str/StringDtype, and not numeric."""
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) \
        and not pd.api.types.is_numeric_dtype(s)


def _is_integer_valued(s: pd.Series) -> bool:
    non_null = s.dropna()
    if non_null.empty:
        return False
    try:
        return bool(np.all(np.mod(non_null.to_numpy(dtype="float64"), 1) == 0))
    except (TypeError, ValueError):
        return False


def _infer_semantic_type(s: pd.Series, n_rows: int) -> SemanticType:
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    try:
        nunique = int(non_null.nunique())
    except TypeError:
        return "text"
    if nunique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        unique_rate = nunique / max(n_rows, 1)
        if unique_rate >= 0.98 and _is_integer_valued(non_null):
            return "id"
        if nunique == 2:
            return "boolean"
        if nunique <= 20 and _is_integer_valued(non_null):
            return "numeric_discrete"
        return "numeric_continuous"
    # text-like
    sample = non_null.sample(min(2000, len(non_null)), random_state=0) if len(non_null) > 2000 else non_null
    lowered = sample.astype(str).str.strip().str.lower()
    if nunique <= 3 and bool(lowered.isin(_BOOL_STRINGS).mean() >= 0.95):
        return "boolean"
    if float(pd.to_numeric(sample, errors="coerce").notna().mean()) >= 0.9:
        coerced = pd.to_numeric(non_null, errors="coerce").dropna()
        if coerced.nunique() <= 20 and _is_integer_valued(coerced):
            return "numeric_discrete"
        return "numeric_continuous"
    if _looks_datetime(sample):
        return "datetime"
    unique_rate = nunique / max(n_rows, 1)
    if unique_rate >= 0.98:
        return "id"
    avg_len = float(sample.astype(str).str.len().mean())
    if nunique > 50 and avg_len > 30:
        return "text"
    return "categorical"


def _looks_datetime(sample: pd.Series) -> bool:
    s = sample.astype(str)
    dateish = s.str.contains(r"[-/:]", regex=True) & (s.str.count(r"\d") >= 4)
    if float(dateish.mean()) < 0.5:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(pd.to_datetime(sample, errors="coerce").notna().mean()) >= 0.9
        except (ValueError, TypeError):
            return False


def _target_association(feat: pd.Series, y: pd.Series, task: TaskType) -> float | None:
    """Cheap monotonic association sketch (|Spearman|). Numeric features only."""
    x = pd.to_numeric(feat, errors="coerce")
    if x.notna().mean() < 0.5:
        return None
    yy = pd.to_numeric(y, errors="coerce") if task == "regression" else _binary_target(y)
    if yy is None:
        return None
    d = pd.DataFrame({"x": x, "y": yy}).dropna()
    if d.shape[0] < 20 or d["x"].nunique() < 2:
        return None
    from scipy import stats
    rho = stats.spearmanr(d["x"], d["y"]).statistic
    return None if rho is None or np.isnan(rho) else round(float(abs(rho)), 4)


def _binary_target(y: pd.Series) -> pd.Series | None:
    vals = pd.Series(y).dropna().unique()
    if len(vals) != 2:
        return None
    classes = sorted(vals, key=str)
    return (pd.Series(y) == classes[-1]).astype(float)


# --------------------------------------------------------------------------- #
# internal profiling
# --------------------------------------------------------------------------- #
def profile_dataframe(
    df: pd.DataFrame, target: str, task: TaskType, config: FeatureAgentConfig
) -> DataProfile:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {list(df.columns)}")
    n_rows = int(len(df))
    y = df[target]
    columns: list[ColumnProfile] = []
    for name in df.columns:
        s = df[name]
        sem = _infer_semantic_type(s, n_rows)
        non_null = s.dropna()
        try:
            n_unique = int(non_null.nunique())
        except TypeError:
            n_unique = int(non_null.astype(str).nunique())
        cp = ColumnProfile(
            name=str(name),
            dtype=str(s.dtype),
            semantic_type=sem,
            n_unique=n_unique,
            unique_rate=round(n_unique / max(n_rows, 1), 4),
            null_rate=round(float(s.isna().mean()), 4),
        )
        if sem in ("numeric_continuous", "numeric_discrete") or (
            pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        ):
            num = pd.to_numeric(non_null, errors="coerce")
            if num.notna().any():
                cp.mean = _r(num.mean()); cp.std = _r(num.std())
                cp.minimum = _r(num.min()); cp.maximum = _r(num.max())
                cp.skew = _r(num.skew())
        if sem in ("categorical", "boolean"):
            vc = non_null.astype(str).value_counts().head(10)
            cp.top_values = [str(k) for k in vc.index.tolist()]
        if not non_null.empty:
            try:
                head = non_null.drop_duplicates().head(5)
            except TypeError:
                head = non_null.head(5)
            cp.sample_values = [str(v)[:60] for v in head.tolist()]
        if name != target and sem in ("numeric_continuous", "numeric_discrete", "boolean"):
            cp.target_association = _target_association(s, y, task)
        columns.append(cp)

    return DataProfile(
        n_rows=n_rows, n_cols=int(df.shape[1]), target=target, task=task,
        source="internal", group_column=config.group_column, columns=columns,
    )


def _r(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)


# --------------------------------------------------------------------------- #
# eda_agent artifact ingestion
# --------------------------------------------------------------------------- #
def ingest_eda_profile(
    report_path: str | Path, target: str, task: TaskType, config: FeatureAgentConfig
) -> DataProfile:
    """Build a DataProfile from an eda_agent `report.json`.

    The EDA agent's column schema is nearly 1:1 with ours (same semantic-type
    vocabulary), so this is a straight mapping. Raises on a malformed file; the
    orchestrator falls back to internal profiling."""
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    prof = payload["analysis"]["profile"]
    assoc: dict[str, float] = {}
    ta = payload["analysis"].get("target_analysis") or {}
    for pair in ta.get("feature_associations", []) or []:
        assoc[str(pair.get("b"))] = abs(float(pair.get("pearson"))) if pair.get("pearson") is not None else None

    columns: list[ColumnProfile] = []
    for c in prof.get("columns", []):
        num = c.get("numeric") or {}
        cat = c.get("categorical") or {}
        columns.append(ColumnProfile(
            name=c["name"], dtype=c.get("dtype", ""),
            semantic_type=c.get("semantic_type", "unknown"),
            n_unique=int(c.get("n_unique", 0)),
            unique_rate=float(c.get("unique_rate", 0.0)),
            null_rate=float(c.get("null_rate", 0.0)),
            mean=num.get("mean"), std=num.get("std"),
            minimum=num.get("minimum"), maximum=num.get("maximum"), skew=num.get("skew"),
            top_values=[str(k) for k in (cat.get("top_values") or {}).keys()][:10],
            sample_values=[str(v)[:60] for v in (c.get("sample_values") or [])[:5]],
            target_association=assoc.get(c["name"]),
        ))
    return DataProfile(
        n_rows=int(prof["n_rows"]), n_cols=int(prof["n_cols"]),
        target=target, task=task, source=f"eda_agent:{report_path}",
        group_column=config.group_column, columns=columns,
    )


def build_profile(
    df: pd.DataFrame, target: str, task: TaskType, config: FeatureAgentConfig,
    eda_report: str | Path | None = None,
) -> DataProfile:
    """Profile the data, preferring an eda_agent artifact when one is supplied and readable."""
    if eda_report and Path(eda_report).exists():
        try:
            return ingest_eda_profile(eda_report, target, task, config)
        except Exception:
            pass  # fall back to internal profiling
    return profile_dataframe(df, target, task, config)
