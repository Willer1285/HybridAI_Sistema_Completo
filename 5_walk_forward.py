# =============================================================
#  PASO 5 - VALIDACIÓN WALK-FORWARD v3.0
#  Solo XAUUSD con 24 features (vol_regime, trend_str, hora).
#  Entrena en ventanas históricas y evalúa en periodos futuros.
#
#  Comando: python 5_walk_forward.py
# =============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"
N_FEATURES      = 24
BARRAS_FUTURO   = 5

# Walk-Forward params
TRAIN_BARS    = 4 * 24192 // 5   # ~3 años de M15
VAL_BARS      = 24192 // 4       # ~3 meses
TEST_BARS     = 24192 // 4       # ~3 meses
STEP_BARS     = 24192 // 4       # Avanza 3 meses

# Modelo v3.0 con regularización fuerte
MODEL_PARAMS = {
    "n_estimators":    500,
    "max_depth":       10,
    "min_samples_leaf": 50,
    "max_features":    "sqrt",
    "n_jobs":          -1,
    "random_state":    42,
}
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE FEATURES v3.0                              ║
# ╚══════════════════════════════════════════════════════════╝

def ema_calc(data, period):
    result = np.full(len(data), np.nan)
    first_valid = 0
    while first_valid < len(data) and np.isnan(data[first_valid]):
        first_valid += 1
    if first_valid + period > len(data):
        return result
    k = 2.0 / (period + 1)
    result[first_valid + period - 1] = np.mean(data[first_valid : first_valid + period])
    for i in range(first_valid + period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1.0 - k)
    return result

def sma_calc(data, period):
    result = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result

def atr_calc(h_, l_, c_, period):
    n = len(c_)
    tr = np.maximum(h_[1:] - l_[1:],
         np.maximum(np.abs(h_[1:] - c_[:-1]),
                    np.abs(l_[1:] - c_[:-1])))
    tr = np.concatenate([[np.nan], tr])
    result = np.full(n, np.nan)
    result[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def calcular_features_v3(df):
    """Calcula los 24 features v3.0 para XAUUSD."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    feats = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

    # RSI(14)
    rsi_p = 14
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n, np.nan); al = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p]); al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs)); rsi[:rsi_p] = np.nan

    # MACD
    ema12 = ema_calc(c, 12); ema26 = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig = ema_calc(macd_line, 9); macd_hist = macd_line - macd_sig

    # ATR
    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    # Bollinger
    sma20 = sma_calc(c, 20)
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19:i+1], ddof=0)
    bb_up = sma20 + 2.0 * std20; bb_lo = sma20 - 2.0 * std20
    bb_w = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp = np.where(sma20 == 0, np.nan, bb_w / sma20)

    # EMAs
    ema9 = ema_calc(c, 9); ema21 = ema_calc(c, 21); ema50 = ema_calc(c, 50)

    def ret(data, p):
        r = np.full(n, np.nan)
        r[p:] = (data[p:] - data[:-p]) / data[:-p] * 100.0
        return r

    vol_ma = sma_calc(v, 20)
    vol_rt = np.where(vol_ma == 0, 1.0, v / vol_ma)

    hl_r = np.where(c == 0, np.nan, (h - l) / c * 100.0)
    rng = h - l
    cl_p = np.where(rng == 0, 0.5, (c - l) / rng)
    bd_r = np.where(rng == 0, 0.0, (c - o) / rng)

    mxh = np.full(n, np.nan); mnl = np.full(n, np.nan)
    for i in range(13, n):
        mxh[i] = np.max(h[i-13:i+1]); mnl[i] = np.min(l[i-13:i+1])
    wl_r = mxh - mnl
    will = np.where(wl_r == 0, 0.5, (c - mnl) / wl_r)

    # Base 20
    feats[:, 0] = rsi/100.0; feats[:, 1] = macd_line/atr_s; feats[:, 2] = macd_sig/atr_s
    feats[:, 3] = macd_hist/atr_s; feats[:, 4] = np.where(c==0, np.nan, atr14/c)
    feats[:, 5] = bb_pctb; feats[:, 6] = bb_bwp
    feats[:, 7] = np.where(ema9==0, np.nan, (c-ema9)/ema9*100.0)
    feats[:, 8] = np.where(ema21==0, np.nan, (c-ema21)/ema21*100.0)
    feats[:, 9] = np.where(ema50==0, np.nan, (c-ema50)/ema50*100.0)
    feats[:, 10] = ret(c,1); feats[:, 11] = ret(c,3); feats[:, 12] = ret(c,5)
    feats[:, 13] = ret(c,10); feats[:, 14] = ret(c,20)
    feats[:, 15] = vol_rt; feats[:, 16] = hl_r; feats[:, 17] = cl_p
    feats[:, 18] = bd_r; feats[:, 19] = will

    # v3.0: 4 nuevos features
    atr_sma50 = sma_calc(atr14, 50)
    feats[:, 20] = np.where((atr_sma50 == 0) | np.isnan(atr_sma50), 1.0, atr14 / atr_sma50)
    feats[:, 21] = np.where(atr_s == 0, 0.0, (ema21 - ema50) / atr_s)

    hours = df.index.hour + df.index.minute / 60.0
    feats[:, 22] = np.sin(2.0 * np.pi * hours / 24.0)
    feats[:, 23] = np.cos(2.0 * np.pi * hours / 24.0)

    return feats, atr14


# ╔══════════════════════════════════════════════════════════╗
# ║  WALK-FORWARD VALIDATION v3.0                            ║
# ╚══════════════════════════════════════════════════════════╝

def walk_forward(df, verbose=True):
    """Ejecuta walk-forward validation para XAUUSD con features v3.0."""
    X_full, atr_vals = calcular_features_v3(df)
    c = df["close"].values

    # Target
    y = np.full(len(c), np.nan)
    y[:-BARRAS_FUTURO] = (
        (c[BARRAS_FUTURO:] - c[:-BARRAS_FUTURO]) / c[:-BARRAS_FUTURO] * 100.0
    )

    # Filtrar NaN
    valid_mask = ~(np.any(np.isnan(X_full), axis=1) | np.isnan(y) | np.isinf(y))

    n_total = valid_mask.sum()
    if n_total < TRAIN_BARS + TEST_BARS:
        if verbose:
            print(f"  Insuficientes datos para walk-forward ({n_total:,} < {TRAIN_BARS + TEST_BARS:,})")
        return []

    X_valid = X_full[valid_mask].astype(np.float32)
    y_valid = np.clip(y[valid_mask], -10.0, 10.0).astype(np.float32)

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

        # Entrenar modelo para esta ventana
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(**MODEL_PARAMS)),
        ])
        pipe.fit(X_tr, y_tr)

        # Evaluar
        y_pred = pipe.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        mask_nz = np.abs(y_te) > 0.01
        dir_acc = np.mean(np.sign(y_pred[mask_nz]) == np.sign(y_te[mask_nz])) * 100 if mask_nz.sum() > 0 else 50.0

        # Umbral adaptativo (percentil 70)
        abs_preds = np.abs(y_pred)
        if len(abs_preds) > 0:
            adaptive_thr = max(np.percentile(abs_preds, 70), 0.005)
        else:
            adaptive_thr = 0.01

        buy_mask = y_pred > adaptive_thr
        sell_mask = y_pred < -adaptive_thr
        buy_ret = y_te[buy_mask]
        sell_ret = -y_te[sell_mask]
        trade_rets = np.concatenate([buy_ret, sell_ret]) if (len(buy_ret) + len(sell_ret)) > 0 else np.array([0])
        n_trades = len(trade_rets)
        mean_ret = np.mean(trade_rets) if n_trades > 0 else 0
        std_ret = np.std(trade_rets) if n_trades > 1 else 1
        sharpe_approx = mean_ret / std_ret * np.sqrt(24192) if std_ret > 0 else 0

        # Overfit check
        y_pred_tr = pipe.predict(X_tr)
        mask_nz_tr = np.abs(y_tr) > 0.01
        dir_tr = np.mean(np.sign(y_pred_tr[mask_nz_tr]) == np.sign(y_tr[mask_nz_tr])) * 100 if mask_nz_tr.sum() > 0 else 50
        overfit_gap = dir_tr - dir_acc

        fold_result = {
            "fold": fold,
            "train_size": int(len(X_tr)),
            "test_size": int(len(X_te)),
            "rmse": round(float(rmse), 5),
            "directional_accuracy": round(float(dir_acc), 2),
            "dir_train": round(float(dir_tr), 2),
            "overfit_gap": round(float(overfit_gap), 2),
            "n_trades": int(n_trades),
            "threshold_used": round(float(adaptive_thr), 5),
            "mean_trade_return": round(float(mean_ret), 4),
            "sharpe_approx": round(float(sharpe_approx), 2),
            "profitable": bool(mean_ret > 0),
        }
        results.append(fold_result)

        if verbose:
            status = "OK" if mean_ret > 0 else "NEG"
            of_flag = " [OF!]" if overfit_gap > 5 else ""
            print(f"    Fold {fold:>2}: Dir={dir_acc:5.1f}%  RMSE={rmse:.4f}  "
                  f"Thr={adaptive_thr:.4f}  Trades={n_trades:>4}  "
                  f"Ret={mean_ret:+.3f}%  Gap={overfit_gap:.1f}pp  [{status}]{of_flag}")

        start += STEP_BARS

    return results


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  WALK-FORWARD VALIDATION v3.0 - XAUUSD ONLY")
    print("=" * 65)
    print(f"\n  Ventana train:  ~{TRAIN_BARS:,} barras (~3 anios)")
    print(f"  Ventana test:   ~{TEST_BARS:,} barras (~3 meses)")
    print(f"  Paso de avance: ~{STEP_BARS:,} barras (~3 meses)")
    print(f"  Modelo: depth={MODEL_PARAMS['max_depth']}, leaf={MODEL_PARAMS['min_samples_leaf']}")

    archivo = f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n  ERROR: {archivo} no encontrado")
        print(f"  -> Ejecuta primero: python 2_descargar_datos.py")
        exit(1)

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD: XAUUSD")
    print(f"{'='*65}")

    df = pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[df["high"] >= df["low"]]

    print(f"  Datos: {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

    results = walk_forward(df)

    if results:
        n_folds = len(results)
        n_profitable = sum(1 for r in results if r["profitable"])
        avg_dir = np.mean([r["directional_accuracy"] for r in results])
        avg_sharpe = np.mean([r["sharpe_approx"] for r in results])
        avg_gap = np.mean([r["overfit_gap"] for r in results])
        consistency = n_profitable / n_folds * 100

        print(f"\n  --- Resumen XAUUSD ---")
        print(f"  Folds totales:       {n_folds}")
        print(f"  Folds rentables:     {n_profitable}/{n_folds} ({consistency:.0f}%)")
        print(f"  Precision media:     {avg_dir:.1f}%")
        print(f"  Sharpe medio:        {avg_sharpe:.2f}")
        print(f"  Overfit gap medio:   {avg_gap:.1f}pp")

        status = "ROBUSTO" if consistency >= 60 and avg_gap < 5 else (
            "ACEPTABLE" if consistency >= 50 else "DEBIL"
        )
        print(f"  Evaluacion:          [{status}]")

        wf_results = {
            "XAUUSD": {
                "folds": results,
                "n_folds": int(n_folds),
                "n_profitable": int(n_profitable),
                "consistency_pct": round(float(consistency), 1),
                "avg_directional_accuracy": round(float(avg_dir), 2),
                "avg_sharpe": round(float(avg_sharpe), 2),
                "avg_overfit_gap": round(float(avg_gap), 2),
                "status": status,
            }
        }
    else:
        wf_results = {"XAUUSD": {"error": "Datos insuficientes"}}

    # Guardar resultados
    wf_path = f"{CARPETA_MODELO}/walk_forward_results.json"
    with open(wf_path, "w") as f:
        json.dump(wf_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {wf_path}")
    print(f"\n  Proximo paso:")
    print(f"    python 6_optimizar_parametros.py")
    print("=" * 65)
