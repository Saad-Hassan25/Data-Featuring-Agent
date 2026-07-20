"""Evaluation harness — marginal lift, not standalone importance (§6.4).

The question is never "is this candidate important among the candidates?" — it is
"does it add signal *beyond what the baseline already has*?" So every candidate is
judged by a paired baseline-vs-augmented cross-validation on identical folds, with
attribution by **out-of-fold permutation importance** and a significance bar set by
**shadow (noise) features** rather than a fixed threshold.

Leakage defense, structural layer: fitted transforms (`group_stat`, `count_encode`,
`bin_quantile`, `target_encode`) are refit inside each fold on *training rows only*
via `FeatureMaterializer` — never on the full frame.

The gradient-boosting model is LightGBM when available (native NaN + categorical
handling) and falls back to scikit-learn's HistGradientBoosting otherwise, so the
harness runs even where LightGBM can't be installed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Cosmetic: LightGBM's sklearn wrapper can emit a feature-name mismatch UserWarning
# when fit on a named frame and scored on a numpy array within the CV loop. The
# numbers are unaffected; silence it so run logs stay readable.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .config import FeatureAgentConfig
from .executor import FeatureMaterializer
from .guards import allowed_columns as _allowed_columns
from .guards import single_feature_score
from .schemas import DataProfile, FeatureSpec

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_MAX_HGB_CATEGORIES = 255  # HistGradientBoosting categorical cap; cap ordinal codes for the fallback


# --------------------------------------------------------------------------- #
# metric plumbing (higher-is-better "orientation" unifies clf & reg)
# --------------------------------------------------------------------------- #
def resolve_metric(task: str, cv_metric: str, n_classes: int) -> str:
    if task == "regression":
        return cv_metric if cv_metric in ("rmse", "mae") else "rmse"
    if n_classes > 2:
        return "accuracy"
    return cv_metric if cv_metric in ("auc", "average_precision") else "auc"


_ORIENT = {"auc": 1, "average_precision": 1, "accuracy": 1, "rmse": -1, "mae": -1}


def compute_metric(name: str, y_true: np.ndarray, score: np.ndarray) -> float:
    if name == "auc":
        return float(roc_auc_score(y_true, score))
    if name == "average_precision":
        return float(average_precision_score(y_true, score))
    if name == "accuracy":
        return float(accuracy_score(y_true, score))
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, score)))
    if name == "mae":
        return float(mean_absolute_error(y_true, score))
    raise ValueError(f"unknown metric '{name}'")


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass
class RoundEvaluation:
    metric_name: str
    orient: int
    baseline_scores: list[float]
    augmented_scores: list[float]
    n_shadows: int
    importances: dict[str, float] = field(default_factory=dict)         # mean OOF perm importance
    per_fold: dict[str, list[float]] = field(default_factory=dict)      # per-fold importance
    shadow_means: list[float] = field(default_factory=list)             # mean importance per shadow
    shadow_ceiling: float = 0.0                                         # config-percentile of shadow_means
    stability: dict[str, float] = field(default_factory=dict)          # sign-consistency across folds
    shadow_percentile: dict[str, float] = field(default_factory=dict)  # candidate percentile vs shadows

    @property
    def baseline_mean(self) -> float:
        return float(np.mean(self.baseline_scores)) if self.baseline_scores else 0.0

    @property
    def baseline_std(self) -> float:
        return float(np.std(self.baseline_scores, ddof=1)) if len(self.baseline_scores) > 1 else 0.0

    @property
    def augmented_mean(self) -> float:
        return float(np.mean(self.augmented_scores)) if self.augmented_scores else 0.0


@dataclass
class ConfirmationResult:
    metric_name: str
    baseline_mean: float
    baseline_std: float
    ablation_mean: float
    ablation_scores: list[float]
    lift: float               # oriented improvement over baseline
    min_lift: float
    passed: bool


@dataclass
class SegmentResult:
    passed: bool
    note: str
    metric_delta: float | None = None
    shadow_percentile: float | None = None


# --------------------------------------------------------------------------- #
# the evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """Owns the base matrix, the CV folds, and the model — reused across rounds."""

    def __init__(self, df: pd.DataFrame, profile: DataProfile, config: FeatureAgentConfig):
        self.config = config
        self.profile = profile
        self.target = profile.target
        self.task = profile.task
        self.group_column = config.group_column
        self.allowed = list(_allowed_columns(profile, config))

        self.df = self._subsample(df)
        self.n = len(self.df)
        self.y_raw = self.df[self.target]
        self.y, self.classes_, self.n_classes = self._encode_target(self.y_raw)
        self.is_binary = self.task == "classification" and self.n_classes == 2
        self.metric_name = resolve_metric(self.task, config.cv_metric, self.n_classes)
        self.orient = _ORIENT[self.metric_name]

        self.X_base, self.base_cols, self.cat_idx = self._build_base()
        self.folds = self._make_folds()

    # -- setup helpers ----------------------------------------------------- #
    def _subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= self.config.max_eval_rows:
            return df.reset_index(drop=True)
        rng = np.random.default_rng(self.config.random_state)
        if self.task == "classification":
            # stratified by target
            idx = []
            frac = self.config.max_eval_rows / len(df)
            for _, grp in df.groupby(df[self.target].astype(str)):
                take = max(1, int(round(len(grp) * frac)))
                idx.extend(rng.choice(grp.index.to_numpy(), size=min(take, len(grp)), replace=False))
            sub = df.loc[idx]
        else:
            sub = df.sample(n=self.config.max_eval_rows, random_state=self.config.random_state)
        return sub.reset_index(drop=True)

    def _encode_target(self, y: pd.Series):
        if self.task == "regression":
            return pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64"), None, 1
        classes = sorted(pd.Series(y).dropna().unique(), key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        return pd.Series(y).map(mapping).to_numpy(), classes, len(classes)

    def _encode_categorical(self, col: pd.Series) -> np.ndarray:
        """Global ordinal encoding (uses no target -> leakage-safe). NaN/unseen -> 0."""
        s = col.astype("string")
        cats = sorted(pd.unique(s.dropna()), key=str)
        mapping = {c: i + 1 for i, c in enumerate(cats)}
        return s.map(mapping).astype("float64").fillna(0.0).to_numpy()

    def _build_base(self):
        data: dict[str, np.ndarray] = {}
        base_cols: list[str] = []
        cat_idx: list[int] = []
        for cp in self.profile.columns:
            name = cp.name
            if name == self.target or name == self.group_column:
                continue
            if name in self.config.forbidden_columns or name in self.config.id_columns:
                continue
            if name not in self.df.columns:
                continue
            st = cp.semantic_type
            col = self.df[name]
            if st in ("numeric_continuous", "numeric_discrete") or (
                pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
            ):
                data[name] = pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
                base_cols.append(name)
            elif st in ("categorical", "boolean"):
                data[name] = self._encode_categorical(col)
                if cp.n_unique <= _MAX_HGB_CATEGORIES:
                    cat_idx.append(len(base_cols))  # declare as categorical to the booster
                base_cols.append(name)
            # id / constant / empty / datetime / text -> excluded from the base matrix
        if base_cols:
            X = np.column_stack([data[c] for c in base_cols])
        else:
            X = np.empty((self.n, 0), dtype="float64")
        return X, base_cols, cat_idx

    def _make_folds(self):
        y = self.y
        groups = self.df[self.group_column].to_numpy() if self.group_column else None
        if self.task == "classification":
            n = min(self.config.n_folds, int(pd.Series(y).value_counts().min()))
        else:
            n = self.config.n_folds
        if groups is not None:
            n = min(n, int(pd.Series(groups).nunique()))
        n = max(2, n)
        rs = self.config.random_state
        if groups is not None:
            if self.task == "classification":
                splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=rs)
                return list(splitter.split(self.X_base, y, groups))
            splitter = GroupKFold(n_splits=n)
            return list(splitter.split(self.X_base, y, groups))
        if self.task == "classification":
            splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=rs)
        else:
            splitter = KFold(n_splits=n, shuffle=True, random_state=rs)
        return list(splitter.split(self.X_base, y))

    # -- model ------------------------------------------------------------- #
    def _new_model(self, n_features: int, cat_idx: list[int]):
        rs = self.config.random_state
        if self.task == "classification":
            if _HAS_LGBM:
                return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                      random_state=rs, n_jobs=1, verbosity=-1,
                                      deterministic=True, force_row_wise=True)
            mask = np.zeros(n_features, dtype=bool)
            for i in cat_idx:
                mask[i] = True
            return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                   random_state=rs, categorical_features=mask)
        if _HAS_LGBM:
            return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                 random_state=rs, n_jobs=1, verbosity=-1,
                                 deterministic=True, force_row_wise=True)
        mask = np.zeros(n_features, dtype=bool)
        for i in cat_idx:
            mask[i] = True
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             random_state=rs, categorical_features=mask)

    def _fit(self, X: np.ndarray, y: np.ndarray, cat_idx: list[int]):
        model = self._new_model(X.shape[1], cat_idx)
        if _HAS_LGBM:
            model.fit(X, y, categorical_feature=cat_idx or "auto")
        else:
            model.fit(X, y)
        return model

    def _predict(self, model, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return model.predict(X)
        if self.is_binary:
            return model.predict_proba(X)[:, 1]
        return model.predict(X)  # multiclass -> labels for accuracy

    def _score(self, y_true: np.ndarray, pred: np.ndarray) -> float:
        return compute_metric(self.metric_name, y_true, pred)

    # -- materialization --------------------------------------------------- #
    def _materialize(self, specs, train_idx, val_idx, y_train):
        mat = FeatureMaterializer(specs=list(specs), allowed_columns=self.allowed)
        df_train = self.df.iloc[train_idx]
        df_val = self.df.iloc[val_idx]
        mat.fit(df_train, y_train)
        return mat.transform(df_train), mat.transform(df_val)

    def _shadows(self, eng_train: pd.DataFrame, eng_val: pd.DataFrame, n_shadows: int, seed: int):
        """S shadow columns: permuted copies of candidate columns (marginal preserved, signal destroyed)."""
        cols = list(eng_train.columns)
        if not cols or n_shadows <= 0:
            empty = pd.DataFrame(index=eng_train.index), pd.DataFrame(index=eng_val.index)
            return empty
        rng = np.random.default_rng(seed)
        st, sv, names = {}, {}, []
        for s in range(n_shadows):
            src = cols[rng.integers(len(cols))]
            nm = f"__shadow_{s}"
            st[nm] = rng.permutation(eng_train[src].to_numpy())
            sv[nm] = rng.permutation(eng_val[src].to_numpy())
            names.append(nm)
        return (pd.DataFrame(st, index=eng_train.index)[names],
                pd.DataFrame(sv, index=eng_val.index)[names])

    # -- the round evaluation --------------------------------------------- #
    def evaluate_candidates(self, specs: list[FeatureSpec],
                            carried: list[FeatureSpec] | None = None) -> RoundEvaluation:
        """Gate `specs` on their marginal lift. `carried` (features already kept in
        earlier rounds) are included in the augmented model so new candidates are
        judged *beyond what we already have* — but they are not themselves gated.

        Baseline is the original-feature model (constant across rounds); attribution
        is out-of-fold permutation importance for `specs` + shadow features."""
        carried = list(carried or [])
        all_specs = list(specs) + carried  # candidate columns first, then carried
        cand_names = [s.name for s in specs]
        n_shadows = self.config.n_shadows(len(specs))
        baseline_scores: list[float] = []
        augmented_scores: list[float] = []
        per_fold: dict[str, list[float]] = {nm: [] for nm in cand_names}
        shadow_fold_imps: list[list[float]] = []

        for f_i, (tr, va) in enumerate(self.folds):
            y_tr, y_va = self.y[tr], self.y[va]
            Xb_tr, Xb_va = self.X_base[tr], self.X_base[va]

            base_model = self._fit(Xb_tr, y_tr, self.cat_idx)  # original features only
            baseline_scores.append(self._score(y_va, self._predict(base_model, Xb_va)))

            if not specs:
                augmented_scores.append(baseline_scores[-1])
                continue

            eng_tr, eng_va = self._materialize(all_specs, tr, va, y_tr)
            cand_tr = eng_tr[cand_names] if cand_names else eng_tr.iloc[:, :0]
            cand_va = eng_va[cand_names] if cand_names else eng_va.iloc[:, :0]
            sh_tr, sh_va = self._shadows(cand_tr, cand_va, n_shadows,
                                         seed=self.config.random_state + 1000 * (f_i + 1))
            sh_cols = list(sh_tr.columns)

            X_tr = np.hstack([Xb_tr, eng_tr.to_numpy(), sh_tr.to_numpy()])
            X_va = np.hstack([Xb_va, eng_va.to_numpy(), sh_va.to_numpy()])
            aug_model = self._fit(X_tr, y_tr, self.cat_idx)  # base is first -> cat indices unchanged
            pred = self._predict(aug_model, X_va)
            augmented_scores.append(self._score(y_va, pred))

            # OOF permutation importance for candidate + shadow columns
            base_oriented = self.orient * self._score(y_va, pred)
            nb = Xb_va.shape[1]
            targets = [(nm, nb + j) for j, nm in enumerate(cand_names)]
            sh_offset = nb + len(all_specs)
            targets += [(nm, sh_offset + s) for s, nm in enumerate(sh_cols)]
            imps = self._perm_importance(aug_model, X_va, y_va, base_oriented, targets, fold=f_i)
            for nm in cand_names:
                per_fold[nm].append(imps[nm])
            shadow_fold_imps.append([imps[nm] for nm in sh_cols])

        return self._assemble(cand_names, baseline_scores, augmented_scores,
                              per_fold, shadow_fold_imps, n_shadows)

    def _perm_importance(self, model, X_va, y_va, base_oriented,
                         targets: list[tuple[str, int]], fold, n_repeats: int = 3) -> dict[str, float]:
        """Drop in oriented score when each target column is permuted in the val fold."""
        out: dict[str, float] = {}
        n_val = X_va.shape[0]
        for k, (nm, col) in enumerate(targets):
            drops = []
            for r in range(n_repeats):
                rng = np.random.default_rng(self.config.random_state + 7 * fold + 31 * k + r)
                Xp = X_va.copy()
                Xp[:, col] = Xp[rng.permutation(n_val), col]
                permuted_oriented = self.orient * self._score(y_va, self._predict(model, Xp))
                drops.append(base_oriented - permuted_oriented)
            out[nm] = float(np.mean(drops))
        return out

    def _assemble(self, cand_names, baseline_scores, augmented_scores,
                  per_fold, shadow_fold_imps, n_shadows) -> RoundEvaluation:
        ev = RoundEvaluation(
            metric_name=self.metric_name, orient=self.orient,
            baseline_scores=baseline_scores, augmented_scores=augmented_scores,
            n_shadows=n_shadows, per_fold=per_fold,
        )
        # mean importance per shadow (over folds), then the config-percentile ceiling
        if shadow_fold_imps:
            n_sh = len(shadow_fold_imps[0])
            shadow_means = [float(np.mean([fold[s] for fold in shadow_fold_imps])) for s in range(n_sh)]
        else:
            shadow_means = []
        ev.shadow_means = shadow_means
        ev.shadow_ceiling = (float(np.percentile(shadow_means, self.config.shadow_percentile))
                             if shadow_means else 0.0)
        for nm in cand_names:
            fold_imps = per_fold.get(nm, [])
            mean_imp = float(np.mean(fold_imps)) if fold_imps else 0.0
            ev.importances[nm] = mean_imp
            ev.stability[nm] = (float(np.mean([1.0 if v > 0 else 0.0 for v in fold_imps]))
                                if fold_imps else 0.0)
            if shadow_means:
                pct = 100.0 * float(np.mean([mean_imp > sm for sm in shadow_means]))
            else:
                pct = 100.0 if mean_imp > 0 else 0.0
            ev.shadow_percentile[nm] = pct
        return ev

    # -- confirmation ablation (§6.4 step 8) ------------------------------- #
    def confirmation(self, kept_specs: list[FeatureSpec]) -> ConfirmationResult:
        baseline_scores, ablation_scores = [], []
        for tr, va in self.folds:
            y_tr, y_va = self.y[tr], self.y[va]
            Xb_tr, Xb_va = self.X_base[tr], self.X_base[va]
            base_model = self._fit(Xb_tr, y_tr, self.cat_idx)
            baseline_scores.append(self._score(y_va, self._predict(base_model, Xb_va)))
            if kept_specs:
                eng_tr, eng_va = self._materialize(kept_specs, tr, va, y_tr)
                X_tr = np.hstack([Xb_tr, eng_tr.to_numpy()])
                X_va = np.hstack([Xb_va, eng_va.to_numpy()])
                model = self._fit(X_tr, y_tr, self.cat_idx)
                ablation_scores.append(self._score(y_va, self._predict(model, X_va)))
            else:
                ablation_scores.append(baseline_scores[-1])
        b_mean = float(np.mean(baseline_scores))
        b_std = float(np.std(baseline_scores, ddof=1)) if len(baseline_scores) > 1 else 0.0
        a_mean = float(np.mean(ablation_scores))
        lift = self.orient * (a_mean - b_mean)
        min_lift = self.config.resolve_min_lift(b_std)
        return ConfirmationResult(
            metric_name=self.metric_name, baseline_mean=b_mean, baseline_std=b_std,
            ablation_mean=a_mean, ablation_scores=ablation_scores,
            lift=lift, min_lift=min_lift, passed=(kept_specs != [] and lift >= min_lift),
        )

    # -- single-feature leakage diagnostic (§6.3 layer 3) ------------------ #
    def single_feature_scores(self, specs: list[FeatureSpec]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        y = pd.Series(self.y, index=self.df.index)
        for spec in specs:
            try:
                mat = FeatureMaterializer(specs=[spec], allowed_columns=self.allowed)
                mat.fit(self.df, y)
                col = mat.transform(self.df)[spec.name]
                out[spec.name] = single_feature_score(col, y, self.task)
            except Exception:
                out[spec.name] = None
        return out

    # -- segment-restricted evaluation (§6.5) ------------------------------ #
    def segment_eval(self, spec: FeatureSpec, segment_expr: str) -> SegmentResult:
        from .executor import validate_expression  # local import avoids cycle at module load
        try:
            validate_expression(segment_expr, set(self.allowed))
        except Exception as exc:
            return SegmentResult(False, f"invalid segment expression: {exc}")
        try:
            mask = self._segment_mask(segment_expr)
        except Exception as exc:
            return SegmentResult(False, f"segment mask failed: {exc}")
        sub = self.df.loc[mask]
        if len(sub) < max(50, 10 * self.config.n_folds):
            return SegmentResult(False, f"segment too small ({len(sub)} rows) for a reliable gate.")
        if self.task == "classification" and pd.Series(self.y[mask.to_numpy()]).nunique() < 2:
            return SegmentResult(False, "segment contains a single class.")
        try:
            sub_eval = Evaluator(sub, self.profile, self.config)
            ev = sub_eval.evaluate_candidates([spec])
        except Exception as exc:
            return SegmentResult(False, f"segment evaluation failed: {exc}")
        pct = ev.shadow_percentile.get(spec.name, 0.0)
        passed = ev.importances.get(spec.name, 0.0) > ev.shadow_ceiling and pct >= self.config.shadow_percentile
        delta = sub_eval.orient * (ev.augmented_mean - ev.baseline_mean)
        return SegmentResult(passed, f"segment n={len(sub)}, shadow_pct={pct:.0f}", delta, pct)

    def _segment_mask(self, expr: str) -> pd.Series:
        """Evaluate a whitelisted boolean expression over raw column values."""
        import ast

        from .executor import _EXPR_FUNCS, validate_expression
        referenced = validate_expression(expr, set(self.allowed))
        ns = {name: self.df[name].to_numpy() for name in referenced}
        ns.update(_EXPR_FUNCS)
        code = compile(ast.parse(expr, mode="eval"), "<segment>", "eval")
        with np.errstate(all="ignore"):
            out = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 — AST-whitelisted
        return pd.Series(np.asarray(out).astype(bool), index=self.df.index)
