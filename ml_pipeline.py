"""
Supply Chain KPI Dashboard — ML Pipeline
=========================================
Faithful Python port of the JavaScript ML module (const ML = ...).

Components:
  1. Feature engineering (lags, rolling stats, deltas, acceleration, calendar, interactions)
  2. Mutual-information feature selection with dormant-feature watchlist
  3. Preprocessing (median imputation, standardization) — fit on train only
  4. Time-series cross-validation splits (expanding window)
  5. Four models: Ridge, XGBoost, Random Forest, MLP
  6. TPE (Bayesian) hyperparameter tuning via Optuna
  7. Nested CV for unbiased model comparison (common outer splits)
  8. Best-model selection per KPI × horizon
  9. Pearson correlations of selected features

Usage:
    from ml_pipeline import forecast_single_kpi

    results = forecast_single_kpi(
        kpi_values=[...],       # list of float
        kpi_dates=["2023-01", "2023-02", ...],
        feature_matrix=[[...], ...],   # n_periods × n_features (or None)
        feature_names=["gdp", "oil_price", ...],
        horizons=[1, 3],
        freq="monthly"
    )
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import optuna

# Silence Optuna logs and sklearn convergence warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing xgboost; fall back gracefully
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ══════════════════════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════════════════════

def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    xa, ya = np.array(x[:n], dtype=float), np.array(y[:n], dtype=float)
    mx, my = xa.mean(), ya.mean()
    dx, dy = xa - mx, ya - my
    denom = math.sqrt(float(np.sum(dx**2) * np.sum(dy**2)))
    if denom == 0:
        return 0.0
    return float(np.sum(dx * dy) / denom)


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (mirrors JS makeSupervised)
# ══════════════════════════════════════════════════════════════

def _lag_policy(n_rows: int) -> dict:
    """Decide lag/rolling window sizes based on series length."""
    if n_rows < 120:
        return dict(feature_lags=[1, 2], kpi_lags=[1, 2], rolling_windows=[2, 3])
    if n_rows < 500:
        return dict(feature_lags=[1, 2, 3, 6], kpi_lags=[1, 2, 3, 6], rolling_windows=[3, 6])
    return dict(feature_lags=[1, 2, 3, 6, 12], kpi_lags=[1, 2, 3, 6, 12], rolling_windows=[3, 6, 12])


def make_supervised(
    kpi_values: list[float],
    feature_matrix: Optional[list[list[float]]],
    feature_names: list[str],
    horizon: int,
    kpi_dates: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Build supervised learning dataset from KPI time series + feature matrix.
    Returns dict with keys: X (np.ndarray), y (np.ndarray), col_names (list[str]).
    """
    n = len(kpi_values)
    if n < 10:
        return None

    lags = _lag_policy(n)
    feature_lags = lags["feature_lags"]
    kpi_lags = lags["kpi_lags"]
    rolling_windows = lags["rolling_windows"]

    cols: list[np.ndarray] = []
    col_names: list[str] = []

    kpi_arr = np.array(kpi_values, dtype=float)

    # ── Raw features + lags + rolling stats + deltas + acceleration ──
    if feature_matrix is not None and len(feature_matrix) == n:
        fm = np.array(feature_matrix, dtype=float)
        p = fm.shape[1]

        # Raw features
        for j in range(p):
            cols.append(fm[:, j])
            col_names.append(feature_names[j] if j < len(feature_names) else f"f{j}")

        # Feature lags
        for j in range(p):
            fname = feature_names[j] if j < len(feature_names) else f"f{j}"
            for lag in feature_lags:
                lagged = np.full(n, np.nan)
                lagged[lag:] = fm[:n - lag, j]
                cols.append(lagged)
                col_names.append(f"{fname}__lag{lag}")

        # Feature rolling stats (mean, std, min, max)
        for j in range(p):
            fname = feature_names[j] if j < len(feature_names) else f"f{j}"
            for w in rolling_windows:
                rm = np.full(n, np.nan)
                rs = np.full(n, np.nan)
                rmin = np.full(n, np.nan)
                rmax = np.full(n, np.nan)
                for i in range(w, n):
                    sl = fm[i - w:i, j]
                    rm[i] = np.mean(sl)
                    rs[i] = np.std(sl, ddof=1) if len(sl) > 1 else 0.0
                    rmin[i] = np.min(sl)
                    rmax[i] = np.max(sl)
                cols.extend([rm, rs, rmin, rmax])
                col_names.extend([
                    f"{fname}__rm{w}", f"{fname}__rs{w}",
                    f"{fname}__rmin{w}", f"{fname}__rmax{w}",
                ])

        # Feature deltas (diff & pct change)
        for j in range(p):
            fname = feature_names[j] if j < len(feature_names) else f"f{j}"
            fdiff = np.full(n, np.nan)
            fpct = np.full(n, np.nan)
            for i in range(1, n):
                prev, cur = fm[i - 1, j], fm[i, j]
                fdiff[i] = cur - prev
                if abs(prev) > 1e-9:
                    fpct[i] = (cur - prev) / abs(prev)
            cols.extend([fdiff, fpct])
            col_names.extend([f"{fname}__delta", f"{fname}__pctchg"])

        # Feature acceleration (diff of diff)
        for j in range(p):
            fname = feature_names[j] if j < len(feature_names) else f"f{j}"
            facc = np.full(n, np.nan)
            for i in range(2, n):
                d1 = fm[i, j] - fm[i - 1, j]
                d0 = fm[i - 1, j] - fm[i - 2, j]
                facc[i] = d1 - d0
            cols.append(facc)
            col_names.append(f"{fname}__accel")

    # ── KPI lags ──
    for lag in kpi_lags:
        lagged = np.full(n, np.nan)
        lagged[lag:] = kpi_arr[:n - lag]
        cols.append(lagged)
        col_names.append(f"kpi__lag{lag}")

    # ── KPI rolling stats ──
    for w in rolling_windows:
        rm = np.full(n, np.nan)
        rs = np.full(n, np.nan)
        rmin = np.full(n, np.nan)
        rmax = np.full(n, np.nan)
        for i in range(w, n):
            sl = kpi_arr[i - w:i]
            rm[i] = np.mean(sl)
            rs[i] = np.std(sl, ddof=1) if len(sl) > 1 else 0.0
            rmin[i] = np.min(sl)
            rmax[i] = np.max(sl)
        cols.extend([rm, rs, rmin, rmax])
        col_names.extend([
            f"kpi__rm{w}", f"kpi__rs{w}", f"kpi__rmin{w}", f"kpi__rmax{w}",
        ])

    # ── KPI momentum & acceleration ──
    diff1 = np.full(n, np.nan)
    diff2 = np.full(n, np.nan)
    for i in range(1, n):
        diff1[i] = kpi_arr[i] - kpi_arr[i - 1]
    for i in range(2, n):
        diff2[i] = diff1[i] - diff1[i - 1]
    cols.extend([diff1, diff2])
    col_names.extend(["kpi__diff1", "kpi__diff2"])

    # ── Calendar features ──
    if kpi_dates and len(kpi_dates) == n:
        month_sin = np.full(n, np.nan)
        month_cos = np.full(n, np.nan)
        qtr = np.full(n, np.nan)
        for i, ds in enumerate(kpi_dates):
            parts = ds.split("-")
            if len(parts) >= 2:
                try:
                    m = int(parts[1])
                    month_sin[i] = math.sin(2 * math.pi * m / 12)
                    month_cos[i] = math.cos(2 * math.pi * m / 12)
                    qtr[i] = math.ceil(m / 3)
                except ValueError:
                    pass
        cols.extend([month_sin, month_cos, qtr])
        col_names.extend(["cal__month_sin", "cal__month_cos", "cal__quarter"])

    # ── Feature interactions (products & ratios, capped at 10) ──
    if feature_matrix is not None and len(feature_matrix) == n:
        fm = np.array(feature_matrix, dtype=float)
        p = fm.shape[1]
        max_interactions = min(10, p * (p - 1) // 2)
        count = 0
        for a in range(p):
            if count >= max_interactions:
                break
            for b in range(a + 1, p):
                if count >= max_interactions:
                    break
                prod = fm[:, a] * fm[:, b]
                ratio = np.full(n, np.nan)
                mask = np.abs(fm[:, b]) > 1e-9
                ratio[mask] = fm[mask, a] / fm[mask, b]
                na = feature_names[a] if a < len(feature_names) else f"f{a}"
                nb = feature_names[b] if b < len(feature_names) else f"f{b}"
                cols.extend([prod, ratio])
                col_names.extend([f"{na}__x__{nb}", f"{na}__div__{nb}"])
                count += 1

    # ── Target: shift by horizon ──
    y_arr = np.full(n, np.nan)
    y_arr[:n - horizon] = kpi_arr[horizon:]

    # ── Build X, y (drop rows with NaN target or >50% NaN features) ──
    all_cols = np.column_stack(cols)  # (n, n_features)
    valid_mask = ~np.isnan(y_arr)
    nan_frac = np.isnan(all_cols).mean(axis=1)
    valid_mask &= nan_frac < 0.5

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 15:
        return None

    X = np.nan_to_num(all_cols[valid_idx], nan=0.0)
    y = y_arr[valid_idx]

    return {"X": X, "y": y, "col_names": col_names}


# ══════════════════════════════════════════════════════════════
# FEATURE SELECTION — MI + dormant feature watchlist
# ══════════════════════════════════════════════════════════════

def select_k_best_mi(X: np.ndarray, y: np.ndarray, k: int) -> dict:
    """
    Select top-k features by mutual information, with dormant feature reactivation.
    Returns dict with 'idx' (selected column indices) and 'X' (filtered array).
    """
    n, p = X.shape
    if k <= 0 or k >= p:
        return {"idx": list(range(p)), "X": X}

    # Step 1: MI-based ranking
    mi_scores = mutual_info_regression(X, y, n_neighbors=3, random_state=42)
    ranked = np.argsort(mi_scores)[::-1]
    selected_set = set(ranked[:k].tolist())
    eliminated = ranked[k:]

    # Step 2: Dormant feature watchlist (z-score > 2 in recent window → reactivate)
    recent_window = max(3, int(n * 0.15))
    WAKE_SIGMA = 2.0
    MAX_REINSERT = max(2, int(k * 0.20))
    reinserted = 0

    for j in eliminated:
        if reinserted >= MAX_REINSERT:
            break
        col = X[:, j]
        hist_end = max(1, n - recent_window)
        hist_vals = col[:hist_end]
        hist_vals = hist_vals[np.isfinite(hist_vals)]
        if len(hist_vals) < 5:
            continue
        hist_mu = np.mean(hist_vals)
        hist_sigma = np.std(hist_vals, ddof=1)

        recent_vals = col[hist_end:]
        recent_vals = recent_vals[np.isfinite(recent_vals)]
        if len(recent_vals) == 0:
            continue
        recent_mu = np.mean(recent_vals)

        if hist_sigma < 1e-12:
            # Near-constant: any change is significant
            if abs(recent_mu - hist_mu) > 1e-9:
                selected_set.add(int(j))
                reinserted += 1
            continue

        z_score = abs(recent_mu - hist_mu) / hist_sigma
        if z_score > WAKE_SIGMA:
            selected_set.add(int(j))
            reinserted += 1

    idx = sorted(selected_set)
    return {"idx": idx, "X": X[:, idx]}


# ══════════════════════════════════════════════════════════════
# PREPROCESSING — fit on train, transform both (no leakage)
# ══════════════════════════════════════════════════════════════

def preprocess_fold(X_train: np.ndarray, X_test: np.ndarray):
    """Impute (median) + standardize. Fit on train only."""
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, imputer, scaler


# ══════════════════════════════════════════════════════════════
# TIME-SERIES CV SPLITS (expanding window, like JS tscvSplits)
# ══════════════════════════════════════════════════════════════

def tscv_splits(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate time-series cross-validation splits (expanding window)."""
    fold_size = n // (n_splits + 1)
    if fold_size < 5:
        # Fallback: single 70/30 split
        cut = int(n * 0.7)
        if cut >= 8 and n - cut >= 3:
            return [(np.arange(cut), np.arange(cut, n))]
        cut = int(n * 0.6)
        if cut >= 6 and n - cut >= 3:
            return [(np.arange(cut), np.arange(cut, n))]
        return []

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_end = (i + 2) * fold_size
        if i == n_splits - 1:
            test_end += n % (n_splits + 1)
        if train_end < 8 or test_end - train_end < 3 or test_end > n:
            continue
        splits.append((np.arange(train_end), np.arange(train_end, test_end)))

    if not splits:
        cut = int(n * 0.7)
        if cut >= 6 and n - cut >= 3:
            return [(np.arange(cut), np.arange(cut, n))]
    return splits


# ══════════════════════════════════════════════════════════════
# AUTO SETTINGS (mirrors JS autoSettings)
# ══════════════════════════════════════════════════════════════

def auto_settings(n_rows: int, n_features: int) -> dict:
    """Compute pipeline settings based on data size."""
    n_splits = 2 if n_rows < 90 else (3 if n_rows < 260 else 5)
    min_rows = max(30, 20) if n_rows >= 80 else max(12, 15)

    ridge_trials = 8 if n_rows < 150 else (10 if n_rows < 500 else 15)
    xgb_trials = 12 if n_rows < 150 else (18 if n_rows < 600 else 25)
    rf_trials = 8 if n_rows < 150 else (12 if n_rows < 500 else 16)
    mlp_trials = 8 if n_rows < 150 else (12 if n_rows < 500 else 16)

    k_best = 0
    if n_features > 0:
        cap_by_n = max(5, int(0.25 * n_rows))
        base = min(n_features, max(10, int(math.sqrt(n_features) * 3)))
        k_best = min(base, cap_by_n, n_features)

    return {
        "n_splits": n_splits,
        "min_rows": min_rows,
        "k_best": k_best,
        "ridge_trials": ridge_trials,
        "xgb_trials": xgb_trials,
        "rf_trials": rf_trials,
        "mlp_trials": mlp_trials,
        "xgb_num_boost_round": 500,
        "xgb_early_stop_rounds": 30,
    }


# ══════════════════════════════════════════════════════════════
# NESTED EVALUATION FUNCTIONS (one per model family)
# ══════════════════════════════════════════════════════════════

def _make_inner_n_splits(cfg: dict) -> int:
    return max(2, cfg["n_splits"] - 1)


def _nested_eval_ridge(X, y, cfg, k_best, outer_splits) -> dict:
    """Nested CV for Ridge: inner TPE tuning + outer evaluation."""
    inner_n_splits = _make_inner_n_splits(cfg)
    maes, rmses = [], []

    for tr_idx, te_idx in outer_splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Inner tuning
        best_alpha = _tune_ridge(X_tr, y_tr, inner_n_splits, cfg["ridge_trials"], k_best)

        # Refit on full train fold
        X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
        sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
        X_tr_fs = sel["X"]
        X_te_fs = X_te_pp[:, sel["idx"]]

        try:
            model = Ridge(alpha=best_alpha)
            model.fit(X_tr_fs, y_tr)
            pred = model.predict(X_te_fs)
            maes.append(mean_absolute_error(y_te, pred))
            rmses.append(math.sqrt(mean_squared_error(y_te, pred)))
        except Exception:
            pass

    return {
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "rmse": float(np.mean(rmses)) if rmses else float("inf"),
    }


def _tune_ridge(X, y, n_splits, n_trials, k_best) -> float:
    """Tune Ridge alpha with inner TSCV + Optuna."""
    inner_splits = tscv_splits(len(X), n_splits)
    if not inner_splits:
        return 1.0

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e7, log=True)
        fold_maes = []
        for tr_idx, te_idx in inner_splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
            sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
            try:
                model = Ridge(alpha=alpha)
                model.fit(sel["X"], y_tr)
                pred = model.predict(X_te_pp[:, sel["idx"]])
                fold_maes.append(mean_absolute_error(y_te, pred))
            except Exception:
                fold_maes.append(1e9)
        return float(np.mean(fold_maes)) if fold_maes else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params.get("alpha", 1.0)


def _nested_eval_xgb(X, y, cfg, k_best, outer_splits) -> dict:
    """Nested CV for XGBoost."""
    if not HAS_XGB:
        return {"mae": float("inf"), "rmse": float("inf")}

    inner_n_splits = _make_inner_n_splits(cfg)
    maes, rmses = [], []

    for tr_idx, te_idx in outer_splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        best_params = _tune_xgb(X_tr, y_tr, inner_n_splits, cfg["xgb_trials"], k_best, cfg)

        X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
        sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
        X_tr_fs = sel["X"]
        X_te_fs = X_te_pp[:, sel["idx"]]

        try:
            # Probe for best n_rounds via early stopping
            n_rounds = best_params.pop("n_estimators", 300)
            cut = max(15, min(int(0.8 * len(X_tr_fs)), len(X_tr_fs) - 5))
            if len(X_tr_fs) >= 20:
                probe = xgb.XGBRegressor(
                    **best_params,
                    n_estimators=cfg["xgb_num_boost_round"],
                    early_stopping_rounds=cfg["xgb_early_stop_rounds"],
                    verbosity=0,
                )
                probe.fit(
                    X_tr_fs[:cut], y_tr[:cut],
                    eval_set=[(X_tr_fs[cut:], y_tr[cut:])],
                    verbose=False,
                )
                n_rounds = probe.best_iteration + 1 if hasattr(probe, "best_iteration") and probe.best_iteration else n_rounds

            # Retrain on full fold
            model = xgb.XGBRegressor(**best_params, n_estimators=n_rounds, verbosity=0)
            model.fit(X_tr_fs, y_tr)
            pred = model.predict(X_te_fs)
            maes.append(mean_absolute_error(y_te, pred))
            rmses.append(math.sqrt(mean_squared_error(y_te, pred)))
        except Exception:
            pass

    return {
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "rmse": float(np.mean(rmses)) if rmses else float("inf"),
    }


def _tune_xgb(X, y, n_splits, n_trials, k_best, cfg) -> dict:
    """Tune XGBoost with inner TSCV + Optuna."""
    inner_splits = tscv_splits(len(X), n_splits)
    if not inner_splits:
        return {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 100}

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.08]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1.0, 3.0, 5.0, 7.0]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0, 5.0]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.1, 0.5]),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
        if params["grow_policy"] == "lossguide":
            params["max_leaves"] = trial.suggest_categorical("max_leaves", [15, 31, 63, 127])
        else:
            params["max_leaves"] = 0

        fold_maes = []
        for tr_idx, te_idx in inner_splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
            sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
            X_tr_fs, X_te_fs = sel["X"], X_te_pp[:, sel["idx"]]

            if len(X_tr_fs) < 25:
                continue

            cut = max(15, min(int(0.8 * len(X_tr_fs)), len(X_tr_fs) - 5))
            try:
                model = xgb.XGBRegressor(
                    **params,
                    n_estimators=cfg["xgb_num_boost_round"],
                    early_stopping_rounds=cfg["xgb_early_stop_rounds"],
                    verbosity=0,
                    tree_method="hist",
                )
                model.fit(
                    X_tr_fs[:cut], y_tr[:cut],
                    eval_set=[(X_tr_fs[cut:], y_tr[cut:])],
                    verbose=False,
                )
                pred = model.predict(X_te_fs)
                fold_maes.append(mean_absolute_error(y_te, pred))
            except Exception:
                fold_maes.append(1e9)

        return float(np.mean(fold_maes)) if fold_maes else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    # Restructure for XGBRegressor
    result = {
        "learning_rate": best.get("learning_rate", 0.05),
        "max_depth": best.get("max_depth", 4),
        "subsample": best.get("subsample", 0.8),
        "colsample_bytree": best.get("colsample_bytree", 0.8),
        "min_child_weight": best.get("min_child_weight", 3.0),
        "reg_lambda": best.get("reg_lambda", 1.0),
        "reg_alpha": best.get("reg_alpha", 0.0),
        "grow_policy": best.get("grow_policy", "depthwise"),
        "max_leaves": best.get("max_leaves", 0),
        "tree_method": "hist",
        "n_estimators": 300,
    }
    return result


def _nested_eval_rf(X, y, cfg, k_best, outer_splits) -> dict:
    """Nested CV for Random Forest."""
    inner_n_splits = _make_inner_n_splits(cfg)
    maes, rmses = [], []

    for tr_idx, te_idx in outer_splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        best_params = _tune_rf(X_tr, y_tr, inner_n_splits, cfg["rf_trials"], k_best)

        X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
        sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
        X_tr_fs = sel["X"]
        X_te_fs = X_te_pp[:, sel["idx"]]

        try:
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_tr_fs, y_tr)
            pred = model.predict(X_te_fs)
            maes.append(mean_absolute_error(y_te, pred))
            rmses.append(math.sqrt(mean_squared_error(y_te, pred)))
        except Exception:
            pass

    return {
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "rmse": float(np.mean(rmses)) if rmses else float("inf"),
    }


def _tune_rf(X, y, n_splits, n_trials, k_best) -> dict:
    """Tune RF with inner TSCV + Optuna."""
    inner_splits = tscv_splits(len(X), n_splits)
    if not inner_splits:
        return {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 5, "max_features": 0.7}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [30, 50, 80]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 12]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [3, 5, 8, 12]),
            "max_features": trial.suggest_categorical("max_features", [0.5, 0.7, 0.85, 1.0]),
        }
        fold_maes = []
        for tr_idx, te_idx in inner_splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
            sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
            try:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(sel["X"], y_tr)
                pred = model.predict(X_te_pp[:, sel["idx"]])
                fold_maes.append(mean_absolute_error(y_te, pred))
            except Exception:
                fold_maes.append(1e9)
        return float(np.mean(fold_maes)) if fold_maes else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params.copy()
    return best


def _nested_eval_mlp(X, y, cfg, k_best, outer_splits) -> dict:
    """Nested CV for MLP."""
    inner_n_splits = _make_inner_n_splits(cfg)
    maes, rmses = [], []

    for tr_idx, te_idx in outer_splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        best_params = _tune_mlp(X_tr, y_tr, inner_n_splits, cfg["mlp_trials"], k_best)

        X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
        sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
        X_tr_fs = sel["X"]
        X_te_fs = X_te_pp[:, sel["idx"]]

        try:
            # Probe for best epochs via early stopping
            hidden = (best_params.get("h1", 32), best_params.get("h2", 16))
            lr = best_params.get("learning_rate_init", 0.001)
            l2 = best_params.get("alpha", 0.001)
            max_iter = best_params.get("max_iter", 200)

            cut = max(12, min(int(0.8 * len(X_tr_fs)), len(X_tr_fs) - 5))
            best_epochs = min(50, max_iter)
            if len(X_tr_fs) >= 17:
                probe = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    learning_rate_init=lr,
                    alpha=l2,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=15,
                    batch_size=min(best_params.get("batch_size", 32), cut),
                    random_state=42,
                )
                probe.fit(X_tr_fs[:cut], y_tr[:cut])
                best_epochs = probe.n_iter_

            # Retrain on full fold
            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                learning_rate_init=lr,
                alpha=l2,
                max_iter=best_epochs,
                early_stopping=False,
                batch_size=min(best_params.get("batch_size", 32), len(X_tr_fs)),
                random_state=42,
            )
            model.fit(X_tr_fs, y_tr)
            pred = model.predict(X_te_fs)
            maes.append(mean_absolute_error(y_te, pred))
            rmses.append(math.sqrt(mean_squared_error(y_te, pred)))
        except Exception:
            pass

    return {
        "mae": float(np.mean(maes)) if maes else float("inf"),
        "rmse": float(np.mean(rmses)) if rmses else float("inf"),
    }


def _tune_mlp(X, y, n_splits, n_trials, k_best) -> dict:
    """Tune MLP with inner TSCV + Optuna."""
    inner_splits = tscv_splits(len(X), n_splits)
    if not inner_splits:
        return {"h1": 32, "h2": 16, "learning_rate_init": 0.001, "alpha": 0.001, "batch_size": 32, "max_iter": 150}

    def objective(trial):
        h1 = trial.suggest_categorical("h1", [16, 32, 48, 64])
        h2 = trial.suggest_categorical("h2", [8, 16, 24, 32])
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003, 0.005])
        l2 = trial.suggest_categorical("l2", [0.0001, 0.001, 0.005, 0.01])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        max_iter = trial.suggest_categorical("max_iter", [100, 150, 200, 300])

        fold_maes = []
        for tr_idx, te_idx in inner_splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            X_tr_pp, X_te_pp, _, _ = preprocess_fold(X_tr, X_te)
            sel = select_k_best_mi(X_tr_pp, y_tr, k_best)
            X_tr_fs, X_te_fs = sel["X"], X_te_pp[:, sel["idx"]]

            cut = max(12, min(int(0.8 * len(X_tr_fs)), len(X_tr_fs) - 5))
            if len(X_tr_fs) < 17:
                continue
            try:
                model = MLPRegressor(
                    hidden_layer_sizes=(h1, h2),
                    learning_rate_init=lr,
                    alpha=l2,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=15,
                    batch_size=min(batch_size, cut),
                    random_state=42,
                )
                model.fit(X_tr_fs[:cut], y_tr[:cut])
                pred = model.predict(X_te_fs)
                fold_maes.append(mean_absolute_error(y_te, pred))
            except Exception:
                fold_maes.append(1e9)

        return float(np.mean(fold_maes)) if fold_maes else 1e9

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    return {
        "h1": best.get("h1", 32),
        "h2": best.get("h2", 16),
        "learning_rate_init": best.get("lr", 0.001),
        "alpha": best.get("l2", 0.001),
        "batch_size": best.get("batch_size", 32),
        "max_iter": best.get("max_iter", 150),
    }


# ══════════════════════════════════════════════════════════════
# MAIN FORECAST PIPELINE FOR A SINGLE KPI
# ══════════════════════════════════════════════════════════════

def forecast_single_kpi(
    kpi_values: list[float],
    kpi_dates: list[str],
    feature_matrix: Optional[list[list[float]]],
    feature_names: list[str],
    horizons: list[int],
    freq: str = "monthly",
) -> list[dict]:
    """
    Full forecast pipeline for one KPI across multiple horizons.

    Returns a list of result dicts, each containing:
      model, horizon, forecast_date, last_value, forecast_value,
      MAE_cv, RMSE_cv, selected_features, is_best
    """
    n = len(kpi_values)
    cfg = auto_settings(n, len(feature_names))
    results = []

    for h in horizons:
        sup = make_supervised(kpi_values, feature_matrix, feature_names, h, kpi_dates)
        if sup is None or len(sup["X"]) < cfg["min_rows"]:
            continue

        X, y, col_names = sup["X"], sup["y"], sup["col_names"]
        k_best = min(cfg["k_best"], X.shape[1])

        # Common outer splits for fair model comparison
        outer_splits = tscv_splits(len(X), cfg["n_splits"])
        if len(outer_splits) < 2:
            continue

        # ── Nested evaluation (all 4 models on SAME outer splits) ──
        try:
            ridge_result = _nested_eval_ridge(X, y, cfg, k_best, outer_splits)
        except Exception:
            ridge_result = {"mae": float("inf"), "rmse": float("inf")}

        try:
            xgb_result = _nested_eval_xgb(X, y, cfg, k_best, outer_splits)
        except Exception:
            xgb_result = {"mae": float("inf"), "rmse": float("inf")}

        try:
            rf_result = _nested_eval_rf(X, y, cfg, k_best, outer_splits)
        except Exception:
            rf_result = {"mae": float("inf"), "rmse": float("inf")}

        try:
            mlp_result = _nested_eval_mlp(X, y, cfg, k_best, outer_splits)
        except Exception:
            mlp_result = {"mae": float("inf"), "rmse": float("inf")}

        # ── Final training on ALL data for actual forecast ──
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_imp)

        inner_n_splits = _make_inner_n_splits(cfg)

        # Forecast date
        fc_date = kpi_dates[-1] if kpi_dates else "?"
        try:
            parts = fc_date.split("-")
            yr, mo = int(parts[0]), int(parts[1])
            if freq == "daily":
                # Approximate
                from datetime import datetime, timedelta
                d = datetime(yr, mo, 1) + timedelta(days=h)
                fc_date = f"{d.year}-{d.month:02d}"
            elif freq == "weekly":
                from datetime import datetime, timedelta
                d = datetime(yr, mo, 1) + timedelta(weeks=h)
                fc_date = f"{d.year}-{d.month:02d}"
            else:
                mo += h
                while mo > 12:
                    mo -= 12
                    yr += 1
                fc_date = f"{yr}-{mo:02d}"
        except Exception:
            pass

        last_value = float(kpi_values[-1])

        # ── Final Ridge ──
        ridge_yhat, ridge_feat_names = float("nan"), []
        try:
            best_alpha = _tune_ridge(X, y, inner_n_splits, cfg["ridge_trials"], k_best)
            sel = select_k_best_mi(X_sc, y, k_best)
            ridge_feat_names = [col_names[j] for j in sel["idx"]]
            model = Ridge(alpha=best_alpha)
            model.fit(sel["X"], y)
            ridge_yhat = float(model.predict(sel["X"][[-1]])[0])
        except Exception:
            pass

        # ── Final XGBoost ──
        xgb_yhat, xgb_feat_names = float("nan"), []
        if HAS_XGB:
            try:
                best_xgb_params = _tune_xgb(X, y, inner_n_splits, cfg["xgb_trials"], k_best, cfg)
                sel = select_k_best_mi(X_sc, y, k_best)
                xgb_feat_names = [col_names[j] for j in sel["idx"]]
                X_fs = sel["X"]

                n_rounds = best_xgb_params.pop("n_estimators", 300)
                cut = max(15, min(int(0.8 * len(X_fs)), len(X_fs) - 5))
                if len(X_fs) >= 20:
                    probe = xgb.XGBRegressor(
                        **best_xgb_params,
                        n_estimators=cfg["xgb_num_boost_round"],
                        early_stopping_rounds=cfg["xgb_early_stop_rounds"],
                        verbosity=0,
                    )
                    probe.fit(X_fs[:cut], y[:cut], eval_set=[(X_fs[cut:], y[cut:])], verbose=False)
                    if hasattr(probe, "best_iteration") and probe.best_iteration:
                        n_rounds = probe.best_iteration + 1

                model = xgb.XGBRegressor(**best_xgb_params, n_estimators=n_rounds, verbosity=0)
                model.fit(X_fs, y)
                xgb_yhat = float(model.predict(X_fs[[-1]])[0])
            except Exception:
                pass

        # ── Final Random Forest ──
        rf_yhat, rf_feat_names = float("nan"), []
        try:
            best_rf_params = _tune_rf(X, y, inner_n_splits, cfg["rf_trials"], k_best)
            sel = select_k_best_mi(X_sc, y, k_best)
            rf_feat_names = [col_names[j] for j in sel["idx"]]
            model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
            model.fit(sel["X"], y)
            rf_yhat = float(model.predict(sel["X"][[-1]])[0])
        except Exception:
            pass

        # ── Final MLP ──
        mlp_yhat, mlp_feat_names = float("nan"), []
        try:
            best_mlp_params = _tune_mlp(X, y, inner_n_splits, cfg["mlp_trials"], k_best)
            sel = select_k_best_mi(X_sc, y, k_best)
            mlp_feat_names = [col_names[j] for j in sel["idx"]]
            X_fs = sel["X"]

            h1 = best_mlp_params.get("h1", 32)
            h2 = best_mlp_params.get("h2", 16)
            lr = best_mlp_params.get("learning_rate_init", 0.001)
            l2 = best_mlp_params.get("alpha", 0.001)
            max_iter = best_mlp_params.get("max_iter", 150)
            bs = best_mlp_params.get("batch_size", 32)

            best_epochs = min(50, max_iter)
            cut = max(12, min(int(0.8 * len(X_fs)), len(X_fs) - 5))
            if len(X_fs) >= 17:
                probe = MLPRegressor(
                    hidden_layer_sizes=(h1, h2),
                    learning_rate_init=lr, alpha=l2,
                    max_iter=max_iter, early_stopping=True,
                    validation_fraction=0.2, n_iter_no_change=15,
                    batch_size=min(bs, cut), random_state=42,
                )
                probe.fit(X_fs[:cut], y[:cut])
                best_epochs = probe.n_iter_

            model = MLPRegressor(
                hidden_layer_sizes=(h1, h2),
                learning_rate_init=lr, alpha=l2,
                max_iter=best_epochs, early_stopping=False,
                batch_size=min(bs, len(X_fs)), random_state=42,
            )
            model.fit(X_fs, y)
            mlp_yhat = float(model.predict(X_fs[[-1]])[0])
        except Exception:
            pass

        # ── Select best model ──
        candidates = [
            {"name": "Ridge",   "yhat": ridge_yhat, "cv": ridge_result, "feat_names": ridge_feat_names},
            {"name": "XGBoost", "yhat": xgb_yhat,   "cv": xgb_result,   "feat_names": xgb_feat_names},
            {"name": "RF",      "yhat": rf_yhat,     "cv": rf_result,    "feat_names": rf_feat_names},
            {"name": "MLP",     "yhat": mlp_yhat,    "cv": mlp_result,   "feat_names": mlp_feat_names},
        ]
        candidates = [c for c in candidates if not math.isnan(c["yhat"]) and c["cv"]["mae"] < float("inf")]

        if not candidates:
            continue

        best_mae = min(c["cv"]["mae"] for c in candidates)

        for c in candidates:
            results.append({
                "model": c["name"],
                "horizon": h,
                "forecast_date": fc_date,
                "last_value": last_value,
                "forecast_value": c["yhat"],
                "MAE_cv": c["cv"]["mae"],
                "RMSE_cv": c["cv"]["rmse"],
                "selected_features": c["feat_names"],
                "is_best": c["cv"]["mae"] == best_mae,
            })

    return results


# ══════════════════════════════════════════════════════════════
# CONVENIENCE: compute correlations from results
# ══════════════════════════════════════════════════════════════

def compute_correlations(
    results: list[dict],
    kpi_values: list[float],
    kpi_dates: list[str],
    feature_matrix: Optional[list[list[float]]],
    feature_names: list[str],
) -> list[dict]:
    """
    For each best-model result, compute Pearson correlation between the KPI
    and each selected feature (raw, before lags).
    """
    corrs = []
    if feature_matrix is None or not feature_names:
        return corrs

    fm = np.array(feature_matrix, dtype=float)
    feat_name_set = {name: j for j, name in enumerate(feature_names)}

    for r in results:
        if not r.get("is_best"):
            continue
        for feat in r.get("selected_features", []):
            # Strip lag/rolling suffixes to find the base feature column
            import re
            base = re.sub(r"__lag\d+|__rm\d+|__rs\d+|__rmin\d+|__rmax\d+|__delta|__pctchg|__accel", "", feat)
            base = re.sub(r"__x__.*|__div__.*", "", base)
            if base in feat_name_set:
                j = feat_name_set[base]
                f_vals = fm[:, j].tolist()
                n = min(len(kpi_values), len(f_vals))
                corr = pearson(kpi_values[:n], f_vals[:n])
                if not math.isnan(corr):
                    corrs.append({
                        "kpi": r.get("kpi", ""),
                        "feature": feat,
                        "forecast_date": r["forecast_date"],
                        "correlation": round(corr, 4),
                    })
    return corrs


# ══════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic monthly data
    n = 60
    dates = [f"{2020 + i // 12}-{(i % 12) + 1:02d}" for i in range(n)]
    kpi = [50 + 0.5 * i + 3 * math.sin(i / 6 * math.pi) + random.gauss(0, 2) for i in range(n)]
    feat1 = [10 + 0.3 * i + random.gauss(0, 1) for i in range(n)]
    feat2 = [kpi[i] * 0.8 + random.gauss(0, 5) for i in range(n)]
    features = [[feat1[i], feat2[i]] for i in range(n)]

    print("Running forecast pipeline on synthetic data (n=60, 2 features, horizons=[1,3])...")
    print("This may take 1-3 minutes due to nested CV + hyperparameter tuning.\n")

    results = forecast_single_kpi(
        kpi_values=kpi,
        kpi_dates=dates,
        feature_matrix=features,
        feature_names=["gdp_index", "demand_proxy"],
        horizons=[1, 3],
        freq="monthly",
    )

    for r in results:
        tag = " ★ BEST" if r["is_best"] else ""
        print(f"  h={r['horizon']}  {r['model']:8s}  "
              f"forecast={r['forecast_value']:.2f}  "
              f"MAE_cv={r['MAE_cv']:.3f}  RMSE_cv={r['RMSE_cv']:.3f}{tag}")

    print(f"\nTotal results: {len(results)}")
    best_results = [r for r in results if r["is_best"]]
    for b in best_results:
        print(f"  → Best for h={b['horizon']}: {b['model']} "
              f"(forecast {b['forecast_value']:.2f}, date {b['forecast_date']})")
        print(f"    Features: {b['selected_features'][:5]}{'...' if len(b['selected_features']) > 5 else ''}")
