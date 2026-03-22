# =============================================================
#  PASO 6 - OPTIMIZACIÓN DE PARÁMETROS v4.0
#  Optimiza hiperparámetros del clasificador y parámetros
#  de trading (SL/TP) para XAUUSD multi-timeframe.
#
#  Comando: python 6_optimizar_parametros.py
# =============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
import itertools
warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"
BARRAS_FUTURO   = 5
ATR_THRESHOLD   = 1.0

MODEL_GRID = {
    "n_estimators": [400, 600],
    "max_depth": [10, 12, 15],
    "min_samples_leaf": [30, 50],
}

TRADING_GRID = {
    "sl_atr_mult": [1.5, 2.0, 2.5],
    "tp_atr_mult": [2.5, 3.0, 3.5, 4.0],
}
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 65)
    print("  OPTIMIZACION v4.0 - CLASIFICACIÓN MULTI-TF")
    print("=" * 65)

    trainer = import_module("3_entrenar_modelo")

    archivos = {
        "m15": f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv",
        "h1":  f"{CARPETA_DATOS}/{SIMBOLO}_h1.csv",
        "h4":  f"{CARPETA_DATOS}/{SIMBOLO}_h4.csv",
        "d1":  f"{CARPETA_DATOS}/{SIMBOLO}_d1.csv",
    }
    for path in archivos.values():
        if not os.path.exists(path):
            print(f"  ERROR: {path} no encontrado"); exit(1)

    dfs = {}
    for tf, path in archivos.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[df["high"] >= df["low"]]
        dfs[tf] = df

    df_m15 = dfs["m15"]

    X_m15, atr_vals = trainer.calcular_features_m15(df_m15)
    X_htf = trainer.calcular_features_htf(df_m15, dfs["h1"], dfs["h4"], dfs["d1"])
    X_full = np.hstack([X_m15, X_htf])

    c = df_m15["close"].values
    y = trainer.crear_target(c, atr_vals, BARRAS_FUTURO, ATR_THRESHOLD)

    valid = (y >= 0) & ~np.any(np.isnan(X_full), axis=1)
    X_v = X_full[valid].astype(np.float32)
    y_v = y[valid]

    # Split 60/20/20
    n = len(X_v)
    idx_train = int(n * 0.60)
    idx_val = int(n * 0.80)

    X_tr, y_tr = X_v[:idx_train], y_v[:idx_train]
    X_val, y_val = X_v[idx_train:idx_val], y_v[idx_train:idx_val]
    X_te, y_te = X_v[idx_val:], y_v[idx_val:]

    print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")

    # Fase 1: Modelo
    print(f"\n  Fase 1: Optimizando modelo...")
    best_score = -1
    best_params = None
    best_pipe = None

    combos = list(itertools.product(
        MODEL_GRID["n_estimators"],
        MODEL_GRID["max_depth"],
        MODEL_GRID["min_samples_leaf"],
    ))

    for i, (n_est, depth, leaf) in enumerate(combos):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesClassifier(
                n_estimators=n_est, max_depth=depth,
                min_samples_leaf=leaf, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=42,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        y_pred_tr = pipe.predict(X_tr)
        acc_tr = accuracy_score(y_tr, y_pred_tr)
        gap = acc_tr - acc
        score = acc - max(0, gap - 0.05) * 0.5

        if score > best_score:
            best_score = score
            best_params = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf}
            best_pipe = pipe

        if (i + 1) % 4 == 0:
            print(f"    {i+1}/{len(combos)} evaluadas...")

    print(f"  Mejor: {best_params} (score={best_score:.4f})")

    # Fase 2: Test
    print(f"\n  Fase 2: Evaluacion test...")
    y_pred_te = best_pipe.predict(X_te)
    acc_te = accuracy_score(y_te, y_pred_te) * 100

    n_buy = np.sum(y_pred_te == 1)
    n_sell = np.sum(y_pred_te == 2)
    buy_prec = np.mean(y_te[y_pred_te == 1] == 1) * 100 if n_buy > 0 else 0
    sell_prec = np.mean(y_te[y_pred_te == 2] == 2) * 100 if n_sell > 0 else 0

    print(f"  Accuracy: {acc_te:.1f}%")
    print(f"  BUY precision: {buy_prec:.1f}% ({n_buy} trades)")
    print(f"  SELL precision: {sell_prec:.1f}% ({n_sell} trades)")

    result = {
        "model_params": best_params,
        "test_accuracy": round(float(acc_te), 2),
        "buy_precision": round(float(buy_prec), 1),
        "sell_precision": round(float(sell_prec), 1),
    }

    opt_path = f"{CARPETA_MODELO}/optimal_params.json"
    with open(opt_path, "w") as f:
        json.dump({"XAUUSD": result}, f, indent=2, ensure_ascii=False)

    print(f"\n  Guardado: {opt_path}")
    print("=" * 65)
