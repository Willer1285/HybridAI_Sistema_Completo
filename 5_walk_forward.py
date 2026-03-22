# =============================================================
#  PASO 5 - WALK-FORWARD v4.0 - CLASIFICACIÓN MULTI-TF
#  Entrena en ventanas temporales y valida con clasificación.
#
#  Comando: python 5_walk_forward.py
# =============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
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

TRAIN_BARS    = 4 * 24192 // 5
TEST_BARS     = 24192 // 4
STEP_BARS     = 24192 // 4

MODEL_PARAMS = {
    "n_estimators":     600,
    "max_depth":        12,
    "min_samples_leaf": 40,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     42,
}
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 65)
    print("  WALK-FORWARD v4.0 - CLASIFICACIÓN MULTI-TF")
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
    print(f"  M15: {len(df_m15):,} barras")

    X_m15, atr_vals = trainer.calcular_features_m15(df_m15)
    X_htf = trainer.calcular_features_htf(df_m15, dfs["h1"], dfs["h4"], dfs["d1"])
    X_full = np.hstack([X_m15, X_htf])

    c = df_m15["close"].values
    y = trainer.crear_target(c, atr_vals, BARRAS_FUTURO, ATR_THRESHOLD)

    valid_mask = (y >= 0) & ~np.any(np.isnan(X_full), axis=1)
    X_valid = X_full[valid_mask].astype(np.float32)
    y_valid = y[valid_mask]

    print(f"  Muestras validas: {len(X_valid):,}")

    results = []
    fold = 0
    start = 0

    while start + TRAIN_BARS + TEST_BARS <= len(X_valid):
        fold += 1
        train_end = start + TRAIN_BARS
        test_end = min(train_end + TEST_BARS, len(X_valid))

        X_tr = X_valid[start:train_end]
        y_tr = y_valid[start:train_end]
        X_te = X_valid[train_end:test_end]
        y_te = y_valid[train_end:test_end]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesClassifier(**MODEL_PARAMS)),
        ])
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred) * 100

        # Balance de predicciones
        n_buy_pred = np.sum(y_pred == 1)
        n_sell_pred = np.sum(y_pred == 2)
        n_neutral_pred = np.sum(y_pred == 0)

        # Simular retornos de trading
        buy_mask = y_pred == 1
        sell_mask = y_pred == 2
        buy_correct = np.mean(y_te[buy_mask] == 1) * 100 if buy_mask.sum() > 0 else 0
        sell_correct = np.mean(y_te[sell_mask] == 2) * 100 if sell_mask.sum() > 0 else 0

        # Train accuracy para detectar overfitting
        y_pred_tr = pipe.predict(X_tr)
        acc_tr = accuracy_score(y_tr, y_pred_tr) * 100

        fold_result = {
            "fold": fold,
            "accuracy_test": round(float(acc), 2),
            "accuracy_train": round(float(acc_tr), 2),
            "overfit_gap": round(float(acc_tr - acc), 2),
            "buy_precision": round(float(buy_correct), 1),
            "sell_precision": round(float(sell_correct), 1),
            "n_buy_pred": int(n_buy_pred),
            "n_sell_pred": int(n_sell_pred),
            "n_neutral_pred": int(n_neutral_pred),
        }
        results.append(fold_result)

        ratio_str = f"B={n_buy_pred} S={n_sell_pred}"
        print(f"    Fold {fold:>2}: Acc={acc:5.1f}%  BuyP={buy_correct:4.1f}%  SellP={sell_correct:4.1f}%  "
              f"Gap={acc_tr-acc:.1f}pp  {ratio_str}")

        start += STEP_BARS

    if results:
        avg_acc = np.mean([r["accuracy_test"] for r in results])
        avg_buy_p = np.mean([r["buy_precision"] for r in results])
        avg_sell_p = np.mean([r["sell_precision"] for r in results])
        avg_gap = np.mean([r["overfit_gap"] for r in results])

        print(f"\n  --- Resumen ---")
        print(f"  Folds: {len(results)}")
        print(f"  Accuracy media: {avg_acc:.1f}%")
        print(f"  Buy precision:  {avg_buy_p:.1f}%")
        print(f"  Sell precision: {avg_sell_p:.1f}%")
        print(f"  Overfit gap:    {avg_gap:.1f}pp")

    wf_path = f"{CARPETA_MODELO}/walk_forward_results.json"
    with open(wf_path, "w") as f:
        json.dump({"XAUUSD": {"folds": results}}, f, indent=2, ensure_ascii=False)

    print(f"\n  Guardado: {wf_path}")
    print("=" * 65)
