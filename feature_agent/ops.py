"""The declarative op library — Tier 1 of the safe executor (§6.2).

Each op is a tiny transformer with `fit`/`transform`, built from a `FeatureSpec`.
Two properties make this the preferred path over free-form code:

  * **Closed vocabulary.** The generation prompt lists exactly these ops and their
    parameter schemas; the LLM composes from them. Nothing else can run.
  * **Known fold-safety class.** Row-wise ops are *stateless* (safe by
    construction). Ops that learn a statistic (`group_stat`, `count_encode`,
    `freq_encode`, `bin_quantile`, `target_encode`) are *fitted* — the executor
    and CV harness guarantee they are fit on training rows only, never the full
    frame, which is the structural leakage defense.

Every op materializes to a single numeric Series, so the augmented modeling
matrix is uniformly numeric (missing values preserved as NaN for the booster to
handle natively).
"""

from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from .guards import InvalidCandidate


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _dt(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _cat(df: pd.DataFrame, col: str) -> pd.Series:
    # stable string key; NaN kept as a distinct <NA> so it forms its own group
    return df[col].astype("string")


def _clean(arr: np.ndarray) -> np.ndarray:
    """Non-finite -> NaN (the downstream booster handles NaN natively).

    Uses np.array (always copies) rather than np.asarray: pandas datetime/boolean
    accessors can back read-only arrays, and we assign into `out` in place."""
    out = np.array(arr, dtype="float64")
    out[~np.isfinite(out)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class BaseOp:
    """Common interface. Subclasses set metadata and implement `_transform`."""

    OP: str = ""               # registry key
    DESC: str = ""             # one-line description for the generation prompt
    INPUTS: str = ""           # human description of expected inputs
    PARAMS: dict[str, str] = {}  # param name -> description (for the prompt)
    FITTED: bool = False       # True -> must be fit on training rows only
    NEEDS_Y: bool = False      # True -> fit consumes the target (target_encode)
    MIN_INPUTS: int = 1
    MAX_INPUTS: int = 1

    def __init__(self, output_name: str, inputs: list[str], params: dict | None = None):
        self.output_name = output_name
        self.inputs = list(inputs)
        self.params = dict(params or {})
        self._state: dict = {}
        self.check()

    # -- validation (build time) ------------------------------------------- #
    def check(self) -> None:
        n = len(self.inputs)
        if not (self.MIN_INPUTS <= n <= self.MAX_INPUTS):
            rng = (f"{self.MIN_INPUTS}" if self.MIN_INPUTS == self.MAX_INPUTS
                   else f"{self.MIN_INPUTS}-{self.MAX_INPUTS}")
            raise InvalidCandidate(f"op '{self.OP}' expects {rng} input(s), got {n}.")
        self._check_params()

    def _check_params(self) -> None:  # override for op-specific param checks
        pass

    def _pfloat(self, key: str, default: float) -> float:
        try:
            return float(self.params.get(key, default))
        except (TypeError, ValueError):
            raise InvalidCandidate(f"op '{self.OP}': param '{key}' must be a number.")

    def _pint(self, key: str, default: int) -> int:
        try:
            return int(self.params.get(key, default))
        except (TypeError, ValueError):
            raise InvalidCandidate(f"op '{self.OP}': param '{key}' must be an integer.")

    # -- fit / transform ---------------------------------------------------- #
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "BaseOp":
        if self.FITTED:
            self._fit(df, y)
        return self

    def _fit(self, df: pd.DataFrame, y: pd.Series | None) -> None:  # fitted ops override
        pass

    def transform(self, df: pd.DataFrame) -> pd.Series:
        values = self._transform(df)
        s = pd.Series(values, index=df.index, name=self.output_name)
        return s

    def _transform(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Tier-1 row-wise (stateless) ops
# --------------------------------------------------------------------------- #
class RatioOp(BaseOp):
    OP, DESC = "ratio", "numerator / (denominator + eps); robust to zero denominators."
    INPUTS = "[numerator, denominator] (two numeric columns)"
    PARAMS = {"eps": "small constant added to the denominator (default 1e-6)"}
    MIN_INPUTS = MAX_INPUTS = 2

    def _transform(self, df):
        eps = self._pfloat("eps", 1e-6)
        num, den = _num(df, self.inputs[0]), _num(df, self.inputs[1])
        return _clean(num.to_numpy() / (den.to_numpy() + eps))


class ProductOp(BaseOp):
    OP, DESC = "product", "element-wise product of two numeric columns."
    INPUTS = "[a, b] (two numeric columns)"
    MIN_INPUTS = MAX_INPUTS = 2

    def _transform(self, df):
        return _clean(_num(df, self.inputs[0]).to_numpy() * _num(df, self.inputs[1]).to_numpy())


class DifferenceOp(BaseOp):
    OP, DESC = "difference", "a - b for two numeric columns."
    INPUTS = "[a, b] (two numeric columns)"
    MIN_INPUTS = MAX_INPUTS = 2

    def _transform(self, df):
        return _clean(_num(df, self.inputs[0]).to_numpy() - _num(df, self.inputs[1]).to_numpy())


class Log1pOp(BaseOp):
    OP, DESC = "log1p", "log(1 + x); values <= -1 become NaN."
    INPUTS = "[x] (one numeric column)"

    def _transform(self, df):
        x = _num(df, self.inputs[0]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(x > -1.0, np.log1p(x), np.nan)
        return _clean(out)


class SqrtOp(BaseOp):
    OP, DESC = "sqrt", "square root of |x| (variance-stabilizing, defined for all reals)."
    INPUTS = "[x] (one numeric column)"

    def _transform(self, df):
        return _clean(np.sqrt(np.abs(_num(df, self.inputs[0]).to_numpy())))


class ClipOp(BaseOp):
    OP, DESC = "clip", "clip x into [lower, upper] (either bound optional)."
    INPUTS = "[x] (one numeric column)"
    PARAMS = {"lower": "lower bound (optional)", "upper": "upper bound (optional)"}

    def _check_params(self):
        if "lower" not in self.params and "upper" not in self.params:
            raise InvalidCandidate("op 'clip' needs at least one of params: lower, upper.")

    def _transform(self, df):
        lower = self._pfloat("lower", -np.inf) if "lower" in self.params else None
        upper = self._pfloat("upper", np.inf) if "upper" in self.params else None
        return _clean(np.clip(_num(df, self.inputs[0]).to_numpy(), lower, upper))


class IsMissingOp(BaseOp):
    OP, DESC = "is_missing", "1 where the column is null, else 0."
    INPUTS = "[x] (any column)"

    def _transform(self, df):
        return df[self.inputs[0]].isna().to_numpy().astype("float64")


class DatePartOp(BaseOp):
    OP, DESC = "date_part", "extract a calendar component from a date column."
    INPUTS = "[date] (one date/datetime column)"
    PARAMS = {"part": "one of: year, quarter, month, day, dayofweek, hour, is_weekend, is_month_end"}
    _PARTS = {"year", "quarter", "month", "day", "dayofweek", "hour", "is_weekend", "is_month_end"}

    def _check_params(self):
        part = self.params.get("part")
        if part not in self._PARTS:
            raise InvalidCandidate(f"op 'date_part': param 'part' must be one of {sorted(self._PARTS)}.")

    def _transform(self, df):
        dt = _dt(df, self.inputs[0]).dt
        part = self.params["part"]
        if part == "is_weekend":
            out = (dt.dayofweek >= 5).astype("float64")
        elif part == "is_month_end":
            out = dt.is_month_end.astype("float64")
        else:
            out = getattr(dt, part).astype("float64")
        return _clean(out.to_numpy())


class DateDiffDaysOp(BaseOp):
    OP, DESC = "date_diff_days", "difference in days between two date columns, or a column and a reference date."
    INPUTS = "[date_a, date_b] OR [date_a] with params.reference"
    PARAMS = {"reference": "ISO date string used as date_b when only one input is given"}
    MIN_INPUTS, MAX_INPUTS = 1, 2

    def _check_params(self):
        if len(self.inputs) == 1 and "reference" not in self.params:
            raise InvalidCandidate("op 'date_diff_days' with one input needs params.reference (ISO date).")

    def _transform(self, df):
        a = _dt(df, self.inputs[0])
        if len(self.inputs) == 2:
            b = _dt(df, self.inputs[1])
        else:
            ref = pd.to_datetime(self.params["reference"], errors="coerce")
            b = pd.Series(ref, index=df.index)
        delta = (a - b).dt.total_seconds() / 86400.0
        return _clean(delta.to_numpy())


class InteractionFlagOp(BaseOp):
    OP, DESC = "interaction_flag", "1 where both conditions hold, else 0 (co-occurrence indicator)."
    INPUTS = "[a, b] (two columns)"
    PARAMS = {"a_value": "value a must equal (optional; default = a is truthy/non-zero)",
              "b_value": "value b must equal (optional; default = b is truthy/non-zero)"}
    MIN_INPUTS = MAX_INPUTS = 2

    def _cond(self, s: pd.Series, key: str) -> np.ndarray:
        if key in self.params:
            return (s.astype("string") == str(self.params[key])).to_numpy()
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.5:  # numeric-ish column -> truthy = non-zero
            return (num.fillna(0) != 0).to_numpy()
        return s.notna().to_numpy()  # otherwise -> truthy = present

    def _transform(self, df):
        a = self._cond(df[self.inputs[0]], "a_value")
        b = self._cond(df[self.inputs[1]], "b_value")
        return (a & b).astype("float64")


# --------------------------------------------------------------------------- #
# Tier-1 fitted (train-only) ops
# --------------------------------------------------------------------------- #
class BinQuantileOp(BaseOp):
    OP, DESC = "bin_quantile", "quantile bin index of x (edges learned on training data)."
    INPUTS = "[x] (one numeric column)"
    PARAMS = {"n_bins": "number of quantile bins (default 5)"}
    FITTED = True

    def _check_params(self):
        if self._pint("n_bins", 5) < 2:
            raise InvalidCandidate("op 'bin_quantile': n_bins must be >= 2.")

    def _fit(self, df, y):
        n_bins = self._pint("n_bins", 5)
        x = _num(df, self.inputs[0]).dropna().to_numpy()
        if x.size == 0:
            self._state["edges"] = np.array([0.0])
            return
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(x, qs))
        self._state["edges"] = edges

    def _transform(self, df):
        edges = self._state.get("edges", np.array([]))
        x = _num(df, self.inputs[0]).to_numpy()
        out = np.digitize(x, edges, right=False).astype("float64")
        out[np.isnan(x)] = np.nan
        return out


class CountEncodeOp(BaseOp):
    OP, DESC = "count_encode", "map each category to its training-set frequency count."
    INPUTS = "[category] (one categorical column)"
    FITTED = True

    def _fit(self, df, y):
        self._state["counts"] = _cat(df, self.inputs[0]).value_counts(dropna=True).to_dict()

    def _transform(self, df):
        counts = self._state.get("counts", {})
        return _cat(df, self.inputs[0]).map(counts).astype("float64").fillna(0.0).to_numpy()


class FreqEncodeOp(BaseOp):
    OP, DESC = "freq_encode", "map each category to its training-set relative frequency."
    INPUTS = "[category] (one categorical column)"
    FITTED = True

    def _fit(self, df, y):
        vc = _cat(df, self.inputs[0]).value_counts(dropna=True, normalize=True)
        self._state["freq"] = vc.to_dict()

    def _transform(self, df):
        freq = self._state.get("freq", {})
        return _cat(df, self.inputs[0]).map(freq).astype("float64").fillna(0.0).to_numpy()


class GroupStatOp(BaseOp):
    OP, DESC = "group_stat", "a numeric column's mean/median/std within a categorical group (learned on train)."
    INPUTS = "[by, value] (categorical group key, numeric value)"
    PARAMS = {"stat": "one of: mean, median, std (default mean)"}
    FITTED = True
    MIN_INPUTS = MAX_INPUTS = 2
    _STATS = {"mean", "median", "std"}

    def _check_params(self):
        if self.params.get("stat", "mean") not in self._STATS:
            raise InvalidCandidate(f"op 'group_stat': param 'stat' must be one of {sorted(self._STATS)}.")

    def _fit(self, df, y):
        stat = self.params.get("stat", "mean")
        by, val = _cat(df, self.inputs[0]), _num(df, self.inputs[1])
        g = pd.DataFrame({"by": by, "v": val}).dropna(subset=["v"])
        table = g.groupby("by", observed=True)["v"].agg(stat)
        self._state["table"] = table.to_dict()
        self._state["global"] = float(getattr(val.dropna(), stat)()) if val.notna().any() else 0.0

    def _transform(self, df):
        table, glob = self._state.get("table", {}), self._state.get("global", 0.0)
        mapped = _cat(df, self.inputs[0]).map(table).astype("float64")
        return _clean(mapped.fillna(glob).to_numpy())


class TargetEncodeOp(BaseOp):
    OP, DESC = "target_encode", "smoothed mean target per category (fold-fitted only; disabled by default)."
    INPUTS = "[category] (one categorical column)"
    PARAMS = {"smoothing": "shrinkage toward the global mean (default 20)"}
    FITTED = True
    NEEDS_Y = True

    def _fit(self, df, y):
        if y is None:
            raise InvalidCandidate("op 'target_encode' requires the target at fit time.")
        yv = pd.to_numeric(pd.Series(np.asarray(y), index=df.index), errors="coerce")
        smoothing = self._pfloat("smoothing", 20.0)
        cats = _cat(df, self.inputs[0])
        d = pd.DataFrame({"c": cats, "y": yv}).dropna(subset=["y"])
        glob = float(d["y"].mean()) if not d.empty else 0.0
        agg = d.groupby("c", observed=True)["y"].agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + glob * smoothing) / (agg["count"] + smoothing)
        self._state["map"] = smooth.to_dict()
        self._state["global"] = glob

    def _transform(self, df):
        m, glob = self._state.get("map", {}), self._state.get("global", 0.0)
        return _clean(_cat(df, self.inputs[0]).map(m).astype("float64").fillna(glob).to_numpy())


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_OP_CLASSES: list[Type[BaseOp]] = [
    RatioOp, ProductOp, DifferenceOp, Log1pOp, SqrtOp, ClipOp, IsMissingOp,
    DatePartOp, DateDiffDaysOp, InteractionFlagOp,
    BinQuantileOp, CountEncodeOp, FreqEncodeOp, GroupStatOp, TargetEncodeOp,
]
OP_REGISTRY: dict[str, Type[BaseOp]] = {c.OP: c for c in _OP_CLASSES}
FITTED_OPS: set[str] = {c.OP for c in _OP_CLASSES if c.FITTED}


def build_op(op_name: str, output_name: str, inputs: list[str], params: dict | None) -> BaseOp:
    cls = OP_REGISTRY.get(op_name)
    if cls is None:
        raise InvalidCandidate(f"unknown op '{op_name}'. Allowed: {sorted(OP_REGISTRY)}")
    return cls(output_name=output_name, inputs=inputs, params=params)


def describe_vocabulary(include_target_encode: bool) -> list[dict]:
    """Machine-readable op catalog for the generation prompt (keeps prompt in sync)."""
    out = []
    for cls in _OP_CLASSES:
        if cls.OP == "target_encode" and not include_target_encode:
            continue
        out.append({
            "op": cls.OP,
            "description": cls.DESC,
            "inputs": cls.INPUTS,
            "params": cls.PARAMS,
            "fold_safety": "fitted (train-only)" if cls.FITTED else "stateless (row-wise)",
        })
    return out
