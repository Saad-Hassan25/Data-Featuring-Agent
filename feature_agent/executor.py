"""The safe executor: specs -> materialized columns (§6.2).

Nothing the LLM emits is ever `eval`'d as free code. Two closed-world tiers:

  * **Tier 1 — declarative ops** (`ops.py`): a fixed vocabulary of transformers.
  * **Tier 2 — row-wise expressions**: arithmetic the op vocabulary can't express
    cleanly, gated by an AST whitelist *before* execution. No attribute access, no
    subscripts, no imports, no dunders, no calls outside a tiny numpy allow-list —
    those node types simply aren't in the whitelist. Expressions evaluate against a
    namespace of only the referenced (non-target) columns plus whitelisted numpy
    functions, so an expression is leakage-safe by construction (row-wise, stateless).

`FeatureMaterializer` is the sklearn-compatible transformer that fits the fitted
ops on training rows and produces the engineered matrix — the same object is used
inside each CV fold and serialized as the export pipeline.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

try:  # sklearn is a core dependency, but keep import failures legible
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "feature_agent requires scikit-learn. Install it with: pip install scikit-learn"
    ) from _exc

from .guards import InvalidCandidate
from .ops import BaseOp, _clean, build_op
from .schemas import FeatureSpec

# --------------------------------------------------------------------------- #
# Tier-2 expression whitelist
# --------------------------------------------------------------------------- #
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Call, ast.Compare,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
)
# Function names resolved to numpy only (never builtins, never attributes).
ALLOWED_FUNCS = {"log1p", "sqrt", "abs", "clip", "where"}
_EXPR_FUNCS = {
    "log1p": np.log1p,
    "sqrt": lambda x: np.sqrt(np.abs(x)),   # variance-stabilizing; defined for all reals
    "abs": np.abs,
    "clip": np.clip,
    "where": np.where,
}


def validate_expression(expr: str, allowed_columns: set[str]) -> set[str]:
    """Validate a row-wise expression; return the set of referenced columns.

    Raises InvalidCandidate for any disallowed node, name, or call form. This is
    the authoritative Tier-2 gate — `allowed_columns` excludes the target and any
    forbidden columns, so a formula that references them fails here (§6.2)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise InvalidCandidate(f"expression does not parse: {exc.msg}")

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise InvalidCandidate(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise InvalidCandidate("disallowed call form (attribute/complex calls not allowed).")
            if node.func.id not in ALLOWED_FUNCS:
                raise InvalidCandidate(f"disallowed function: {node.func.id}")
            if node.keywords:
                raise InvalidCandidate("keyword arguments are not allowed in expressions.")
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id not in allowed_columns:
                raise InvalidCandidate(f"unknown name: {node.id}")
            referenced.add(node.id)
    if not referenced:
        raise InvalidCandidate("expression references no columns.")
    return referenced


class ExpressionOp(BaseOp):
    """A validated Tier-2 expression as a stateless op."""

    OP = "expression"
    FITTED = False

    def __init__(self, output_name: str, expression: str, allowed_columns: set[str]):
        self.output_name = output_name
        self.expression = expression
        self.params = {}
        self._state = {}
        self._referenced = validate_expression(expression, set(allowed_columns))
        self.inputs = sorted(self._referenced)
        self._code = compile(ast.parse(expression, mode="eval"), "<feature-expr>", "eval")

    def check(self) -> None:  # validation happened in __init__
        pass

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        ns = {name: pd.to_numeric(df[name], errors="coerce").to_numpy() for name in self._referenced}
        ns.update(_EXPR_FUNCS)
        with np.errstate(all="ignore"):
            out = eval(self._code, {"__builtins__": {}}, ns)  # noqa: S307 — AST-whitelisted, no builtins
        arr = np.asarray(out, dtype="float64")
        if arr.ndim == 0 or arr.shape != (len(df),):
            arr = np.broadcast_to(arr, (len(df),)).astype("float64")
        return _clean(arr)


# --------------------------------------------------------------------------- #
# spec -> op
# --------------------------------------------------------------------------- #
def build_feature(spec: FeatureSpec, allowed_columns: set[str]) -> BaseOp:
    """Construct the op for one spec. Raises InvalidCandidate on any problem."""
    if spec.kind == "expression":
        return ExpressionOp(spec.name, spec.expression or "", allowed_columns)
    return build_op(spec.op or "", spec.name, spec.inputs, spec.params)


def dry_run(spec: FeatureSpec, df: pd.DataFrame, allowed_columns: set[str],
            y: pd.Series | None = None) -> None:
    """Fit+transform a single spec on the full frame to prove it executes.

    Used at the validate/materialize stage so runtime failures (unparseable
    dates, empty output, op errors) are rejected with a structured reason instead
    of exploding inside the CV loop. Raises InvalidCandidate on failure."""
    op = build_feature(spec, allowed_columns)
    try:
        op.fit(df, y)
        out = op.transform(df)
    except InvalidCandidate:
        raise
    except Exception as exc:  # any runtime failure -> structured rejection
        raise InvalidCandidate(f"materialization failed: {type(exc).__name__}: {exc}")
    if out.notna().sum() == 0:
        raise InvalidCandidate("materialized to all-null (no usable values).")


# --------------------------------------------------------------------------- #
# FeatureMaterializer (sklearn transformer; also the export pipeline step)
# --------------------------------------------------------------------------- #
class FeatureMaterializer(BaseEstimator, TransformerMixin):
    """Build a DataFrame of engineered features from a list of specs.

    Stateless ops are safe anywhere; fitted ops (`group_stat`, `count_encode`,
    `bin_quantile`, `target_encode`, ...) learn their statistics in `fit` and are
    thus fit on whatever rows `fit` is given — the CV harness passes training-fold
    rows only, and the export pipeline refits on the full training frame.
    """

    def __init__(self, specs: list[FeatureSpec] | None = None,
                 allowed_columns: list[str] | None = None):
        self.specs = specs or []
        self.allowed_columns = allowed_columns or []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureMaterializer":
        allowed = set(self.allowed_columns)
        y_series = None if y is None else pd.Series(np.asarray(y), index=X.index)
        self.ops_: list[BaseOp] = []
        for spec in self.specs:
            op = build_feature(spec, allowed)
            op.fit(X, y_series)
            self.ops_.append(op)
        self.feature_names_ = [op.output_name for op in self.ops_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "ops_", None):
            return pd.DataFrame(index=X.index)
        cols = {op.output_name: op.transform(X) for op in self.ops_}
        return pd.DataFrame(cols, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "feature_names_", []), dtype=object)
