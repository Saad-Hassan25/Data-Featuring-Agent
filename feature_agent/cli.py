"""Command-line interface.

    python -m feature_agent data.csv --target churned --task classification \\
        --forbidden cancellation_date --group-column customer_id \\
        --domain "B2B SaaS; churn = no renewal within 30 days"
    python -m feature_agent --demo                    # reference churn scenario
    python -m feature_agent data.csv --target y --task regression --no-llm

Configuration precedence: CLI flags > environment / .env > defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import FeatureAgentConfig
from .orchestrator import FeatureAgent


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    sfx = p.suffix.lower()
    if sfx in {".csv", ".txt"}:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(p, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(p)
    if sfx in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if sfx in {".json"}:
        return pd.read_json(p)
    if sfx in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if sfx in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type '{sfx}'.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="feature_agent",
        description="Agentic Feature Engineering & Selection (OpenRouter-backed).")
    ap.add_argument("data", nargs="?", help="Path to a dataset (.csv/.parquet/.json/.xlsx).")
    ap.add_argument("--demo", action="store_true", help="Run on the built-in churn scenario.")
    ap.add_argument("--target", help="Target column.")
    ap.add_argument("--task", choices=["classification", "regression"], help="Problem type.")
    ap.add_argument("--domain", default="", help="Domain context for the generator.")
    ap.add_argument("--group-column", help="Entity id column for GroupKFold.")
    ap.add_argument("--forbidden", help="Comma-separated post-outcome/leaky columns to forbid.")
    ap.add_argument("--eda-report", help="Path to an eda_agent report.json to ingest as the profile.")
    ap.add_argument("--model", help="OpenRouter model id (overrides OPENROUTER_MODEL).")
    ap.add_argument("--max-rounds", type=int, help="Max generation/evaluation rounds.")
    ap.add_argument("--n-candidates", type=int, help="Candidates proposed per round.")
    ap.add_argument("--max-cost", type=float, help="Hard cost ceiling in USD.")
    ap.add_argument("--output", help="Output directory (default: feature_runs).")
    ap.add_argument("--no-llm", action="store_true", help="Deterministic generator/report (no API).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data and not args.demo:
        build_parser().print_help()
        print("\nNothing to do. Provide a data file or use --demo.", file=sys.stderr)
        return 2

    forbidden = [c.strip() for c in (args.forbidden or "").split(",") if c.strip()]
    cfg = FeatureAgentConfig.from_env(
        model=args.model, max_rounds=args.max_rounds, n_candidates_per_round=args.n_candidates,
        max_cost_usd=args.max_cost, group_column=args.group_column,
        forbidden_columns=forbidden or None, output_dir=args.output,
        use_llm=not args.no_llm,
    )

    if args.demo:
        from .sample_data import make_churn_sample
        df = make_churn_sample()
        target = args.target or "churned"
        task = args.task or "classification"
        domain = args.domain or "B2B SaaS subscription business; churn = no renewal within 30 days."
        if not cfg.group_column:
            cfg.group_column = "customer_id"
        if "cancellation_date" not in cfg.forbidden_columns:
            cfg.forbidden_columns.append("cancellation_date")
    else:
        if not args.target or not args.task:
            print("Error: --target and --task are required for real data.", file=sys.stderr)
            return 2
        try:
            df = _load(args.data)
        except Exception as exc:
            print(f"Failed to load '{args.data}': {exc}", file=sys.stderr)
            return 1
        target, task, domain = args.target, args.task, args.domain

    if df.empty:
        print("Loaded dataset is empty.", file=sys.stderr)
        return 1

    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.", file=sys.stderr)
    try:
        result = FeatureAgent(cfg).run(df, target=target, task=task,
                                       domain_context=domain, eda_report=args.eda_report)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nKept {len(result.kept_features)} feature(s); "
          f"{result.metric_name} {result.baseline_metric:.4f} -> {result.final_metric:.4f} "
          f"({result.lift:+.4f}). Cost ${result.cost_usd:.4f}.", file=sys.stderr)
    print(f"  Report:   {result.report_path}", file=sys.stderr)
    print(f"  Run dir:  {result.run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
