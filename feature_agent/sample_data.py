"""Synthetic datasets: the reference churn scenario (§14) and planted-signal data.

Living inside the package means `feature-agent --demo` works when installed, and
the tests get a network-free ground truth.

Design note: the baseline model is a gradient-boosted tree, which already
reconstructs many monotonic transforms and interactions of the *raw* columns — so
a fair "planted signal the agent should recover" is one the baseline genuinely
cannot see. Raw date columns are excluded from the base matrix (only reachable via
date ops), so a date-derived effect is exactly such a signal: it is real, it is
not a target proxy, and an engineered `date_part` feature is the only way to
capture it. Both datasets below plant that kind of signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_churn_sample(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """B2B SaaS churn. Churn is driven by a tickets-per-spend ratio, login recency,
    an enterprise-tier × recency interaction, and a signup-month cohort effect
    (holiday-cohort accounts churn more) — the last of which the baseline cannot
    see because raw dates are excluded from the model matrix. `cancellation_date`
    is a post-outcome leak (declare it in forbidden_columns)."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["free", "pro", "enterprise"], size=n, p=[0.6, 0.3, 0.1])
    base_spend = np.where(plan == "free", 20, np.where(plan == "pro", 120, 800)).astype("float64")
    monthly_spend = np.abs(base_spend * rng.lognormal(0, 0.4, n))
    support_tickets_90d = rng.poisson(np.where(plan == "enterprise", 4, 2), n).astype("float64")
    days_since_login = rng.exponential(20, n).clip(0, 180)
    num_logins_30d = rng.poisson(8, n).astype("float64")
    account_tenure_days = rng.integers(30, 1500, n).astype("float64")
    region = rng.choice(["NA", "EU", "APAC", "LATAM"], size=n, p=[0.45, 0.3, 0.15, 0.1])

    signup = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        (rng.integers(0, 1200, n)).astype(int), unit="D")
    signup_month = signup.month.to_numpy()
    holiday_cohort = np.isin(signup_month, [11, 12, 1]).astype("float64")

    is_ent = (plan == "enterprise").astype("float64")
    tickets_per_spend = support_tickets_90d / (monthly_spend + 1.0)
    logit = (
        -2.3
        + 6.0 * tickets_per_spend                     # ratio signal
        + 0.015 * days_since_login                     # recency
        + 0.9 * is_ent * (days_since_login > 40)       # enterprise × recency
        + 1.0 * holiday_cohort                          # signup-month cohort (date-only signal)
        - 0.03 * num_logins_30d
        + rng.normal(0, 0.3, n)
    )
    churned = (rng.random(n) < _sigmoid(logit)).astype(int)

    cancel_offset = pd.to_timedelta((rng.integers(1, 60, n)).astype(int), unit="D")
    cancellation_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    churn_mask = churned.astype(bool)
    cancellation_date[churn_mask] = (signup[churn_mask] + cancel_offset[churn_mask])

    return pd.DataFrame({
        "customer_id": np.arange(100000, 100000 + n),
        "plan_tier": plan,
        "monthly_spend": monthly_spend.round(2),
        "support_tickets_90d": support_tickets_90d.astype(int),
        "days_since_login": days_since_login.round(1),
        "num_logins_30d": num_logins_30d.astype(int),
        "account_tenure_days": account_tenure_days.astype(int),
        "region": region,
        "signup_date": signup.astype(str),
        "cancellation_date": cancellation_date,
        "churned": churned,
    })


def add_leak_columns(df: pd.DataFrame, target: str, seed: int = 0) -> pd.DataFrame:
    """Append two leak traps for the canary tests:
      * `account_closed`  — an exact copy of the target (declare forbidden).
      * `churn_risk_proxy` — a ~98%-correlated proxy (NOT declared; must end flagged).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    y = out[target].to_numpy()
    out["account_closed"] = y
    flip = rng.random(len(out)) < 0.02
    out["churn_risk_proxy"] = np.where(flip, 1 - y, y).astype(float)
    return out


def make_planted_regression(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A regression target with a planted, base-invisible date signal.

    target = 5*is_weekend(event_date) + 3*sin(2π*month/12) + 0.5*z1 + noise.

    `event_date` is excluded from the base model matrix, so the only way to
    capture the weekend/seasonal effect is an engineered `date_part` feature — the
    ground truth the agent should recover. Columns z1..z5 are decoys."""
    rng = np.random.default_rng(seed)
    event_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n).astype(int), unit="D")
    dow = event_date.dayofweek.to_numpy()
    month = event_date.month.to_numpy()
    is_weekend = (dow >= 5).astype("float64")
    decoys = {f"z{i}": rng.normal(0, 1, n) for i in range(1, 6)}
    target = (5.0 * is_weekend + 3.0 * np.sin(2 * np.pi * month / 12.0)
              + 0.5 * decoys["z1"] + rng.normal(0, 1.5, n))
    return pd.DataFrame({"event_date": event_date.astype(str), **decoys, "target": target})
