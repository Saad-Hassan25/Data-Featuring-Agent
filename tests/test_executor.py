"""Adversarial executor suite (§10). Every unsafe expression must raise
InvalidCandidate and never execute; valid expressions and ops materialize."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_agent.executor import ExpressionOp, build_feature, dry_run, validate_expression
from feature_agent.guards import InvalidCandidate
from feature_agent.schemas import FeatureSpec

ALLOWED = {"a", "b", "monthly_spend", "days_since_login"}

ATTACKS = [
    "__import__('os')",                 # dunder / import
    "os.system('rm -rf /')",            # attribute + unknown name
    "a.__class__",                      # attribute access
    "data['a']",                        # subscript
    "a['x']",                           # subscript on a column
    "open('secret')",                   # disallowed function
    "eval('1')",                        # disallowed function
    "churned + 1",                      # reference to a non-allowed (target) column
    "unknown_col * 2",                  # unknown name
    "lambda x: x",                      # lambda node
    "a if b else 0",                    # IfExp node
    "[a for a in b]",                   # comprehension
    "a.mean()",                         # method call via attribute
    "sqrt(a).real",                     # attribute on a call
]


@pytest.mark.parametrize("expr", ATTACKS)
def test_malicious_expressions_rejected(expr):
    with pytest.raises(InvalidCandidate):
        validate_expression(expr, ALLOWED)


@pytest.mark.parametrize("expr", ATTACKS)
def test_malicious_expressions_never_build(expr):
    # ExpressionOp validates in __init__, before any compile/eval can happen.
    with pytest.raises(InvalidCandidate):
        ExpressionOp("bad_feature", expr, ALLOWED)


def test_valid_expression_returns_referenced_columns():
    refs = validate_expression("monthly_spend / (days_since_login + 1)", ALLOWED)
    assert refs == {"monthly_spend", "days_since_login"}


def test_valid_expression_executes():
    df = pd.DataFrame({"monthly_spend": [100.0, 200.0, np.nan],
                       "days_since_login": [9.0, 0.0, 4.0]})
    op = ExpressionOp("spd", "monthly_spend / (days_since_login + 1)", ALLOWED)
    out = op.fit(df).transform(df)
    assert list(np.round(out.to_numpy(), 4)[:2]) == [10.0, 200.0]
    assert np.isnan(out.to_numpy()[2])  # NaN propagates, no crash


def test_allowed_functions_only():
    # whitelisted numpy funcs are fine; anything else is rejected
    validate_expression("log1p(abs(a)) + sqrt(b)", ALLOWED)
    with pytest.raises(InvalidCandidate):
        validate_expression("pow(a, b)", ALLOWED)  # 'pow' not in ALLOWED_FUNCS


def test_dry_run_rejects_all_null_output():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    spec = FeatureSpec(name="allnull", kind="op", op="log1p", inputs=["a"])
    with pytest.raises(InvalidCandidate):
        dry_run(spec, df, {"a", "b"})


def test_expression_no_columns_rejected():
    with pytest.raises(InvalidCandidate):
        validate_expression("1 + 2", ALLOWED)
