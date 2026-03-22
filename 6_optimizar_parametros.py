# =============================================================
#  PASO 6 - OPTIMIZACIÓN DE PARÁMETROS v3.0
#  Solo XAUUSD. Busca la mejor combinación de modelo +
#  parámetros de trading usando validación temporal.
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

# Grid de hiperparámetros del modelo
MODEL_GRID = {
    "n_estimators": [400, 500, 600],
    "max_depth": [8, 10, 12],
    "min_samples_leaf": [30, 50, 80],
}

# Grid de parámetros de trading
TRADING_GRID = {
    "threshold": [0.05, 0.10, 0.15, 0.20, 0.30],
    "sl_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "tp_atr_mult": [2.5, 3.0, 3.5, 4.0, 5.0],
}

# Costo XAUUSD
SPREAD_COST_PCT = 0.015   # ~$0.30 / $2000 = 0.015%
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
# ║  EVALUACIÓN RÁPIDA DE TRADING                            ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_trading_params(y_true, y_pred, atr_vals, threshold, sl_mult, tp_mult, spread_pct):
    """Evalúa una combinación de parámetros de trading."""
    buy_mask = y_pred > threshold
    sell_mask = y_pred < -threshold

    if buy_mask.sum() + sell_mask.sum() < 10:
        return {"sharpe": -999, "n_trades": 0, "return_pct": 0, "win_rate": 0}

    buy_returns = []
    for idx in np.where(buy_mask)[0]:
        actual_ret = y_true[idx]
        atr_pct = atr_vals[idx] * 100 if not np.isnan(atr_vals[idx]) else 0.5
        sl_ret = -sl_mult * atr_pct
        tp_ret = tp_mult * atr_pct
        clipped = np.clip(actual_ret, sl_ret, tp_ret)
        buy_returns.append(clipped - spread_pct)

    sell_returns = []
    for idx in np.where(sell_mask)[0]:
        actual_ret = -y_true[idx]
        atr_pct = atr_vals[idx] * 100 if not np.isnan(atr_vals[idx]) else 0.5
        sl_ret = -sl_mult * atr_pct
        tp_ret = tp_mult * atr_pct
        clipped = np.clip(actual_ret, sl_ret, tp_ret)
        sell_returns.append(clipped - spread_pct)

    all_returns = np.array(buy_returns + sell_returns)
    n_trades = len(all_returns)
    if n_trades < 10:
        return {"sharpe": -999, "n_trades": 0, "return_pct": 0, "win_rate": 0}

    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    win_rate = np.mean(all_returns > 0) * 100
    sharpe = mean_ret / std_ret * np.sqrt(24192) if std_ret > 0 else 0
    total_return = np.sum(all_returns)

    return {
        "sharpe": round(float(sharpe), 2),
        "n_trades": n_trades,
        "return_pct": round(float(total_return), 2),
        "win_rate": round(float(win_rate), 1),
        "mean_trade_pct": round(float(mean_ret), 4),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  OPTIMIZACIÓN XAUUSD                                     ║
# ╚══════════════════════════════════════════════════════════╝

def optimize_xauusd(df, verbose=True):
    """Optimiza modelo + parámetros de trading para XAUUSD."""
    X_full, atr_vals = calcular_features_v3(df)
    c = df["close"].values

    y = np.full(len(c), np.nan)
    y[:-BARRAS_FUTURO] = (c[BARRAS_FUTURO:] - c[:-BARRAS_FUTURO]) / c[:-BARRAS_FUTURO] * 100.0

    valid = ~(np.any(np.isnan(X_full), axis=1) | np.isnan(y) | np.isinf(y))
    X_v = X_full[valid].astype(np.float32)
    y_v = np.clip(y[valid], -10.0, 10.0).astype(np.float32)
    atr_v = np.where(c[valid] == 0, np.nan, atr_vals[valid] / c[valid])

    # Split: 60% train, 20% val (param search), 20% test (final eval)
    n = len(X_v)
    idx_train = int(n * 0.60)
    idx_val = int(n * 0.80)

    X_tr, y_tr = X_v[:idx_train], y_v[:idx_train]
    X_val, y_val = X_v[idx_train:idx_val], y_v[idx_train:idx_val]
    X_te, y_te = X_v[idx_val:], y_v[idx_val:]
    atr_val = atr_v[idx_train:idx_val]
    atr_te = atr_v[idx_val:]

    if verbose:
        print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")

    # ── FASE 1: Optimizar hiperparámetros del modelo ──
    if verbose:
        print(f"\n  Fase 1: Optimizando hiperparametros del modelo...")

    best_model_score = -999
    best_model_params = None
    best_pipeline = None

    model_combos = list(itertools.product(
        MODEL_GRID["n_estimators"],
        MODEL_GRID["max_depth"],
        MODEL_GRID["min_samples_leaf"],
    ))

    for i, (n_est, depth, leaf) in enumerate(model_combos):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=n_est, max_depth=depth,
                min_samples_leaf=leaf, max_features="sqrt",
                n_jobs=-1, random_state=42,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred_val = pipe.predict(X_val)

        mask_nz = np.abs(y_val) > 0.01
        dir_acc = np.mean(np.sign(y_pred_val[mask_nz]) == np.sign(y_val[mask_nz])) * 100 if mask_nz.sum() > 0 else 50

        # Penalizar overfitting
        y_pred_tr = pipe.predict(X_tr)
        mask_nz_tr = np.abs(y_tr) > 0.01
        dir_tr = np.mean(np.sign(y_pred_tr[mask_nz_tr]) == np.sign(y_tr[mask_nz_tr])) * 100 if mask_nz_tr.sum() > 0 else 50
        overfit_penalty = max(0, (dir_tr - dir_acc) - 3) * 0.5  # Penalizar gap > 3pp
        score = dir_acc - overfit_penalty

        if score > best_model_score:
            best_model_score = score
            best_model_params = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf}
            best_pipeline = pipe

        if verbose and (i + 1) % 9 == 0:
            print(f"    {i+1}/{len(model_combos)} combinaciones evaluadas...")

    if verbose:
        print(f"  Mejor modelo: {best_model_params} (score={best_model_score:.1f})")

    # ── FASE 2: Optimizar parámetros de trading ──
    if verbose:
        print(f"\n  Fase 2: Optimizando parametros de trading...")

    y_pred_val = best_pipeline.predict(X_val)

    best_trading_score = -999
    best_trading_params = None

    trading_combos = list(itertools.product(
        TRADING_GRID["threshold"],
        TRADING_GRID["sl_atr_mult"],
        TRADING_GRID["tp_atr_mult"],
    ))

    for thr, sl_m, tp_m in trading_combos:
        if tp_m <= sl_m:
            continue

        result = evaluate_trading_params(y_val, y_pred_val, atr_val, thr, sl_m, tp_m, SPREAD_COST_PCT)

        if result["sharpe"] > best_trading_score and result["n_trades"] >= 20:
            best_trading_score = result["sharpe"]
            best_trading_params = {
                "threshold": thr,
                "sl_atr_mult": sl_m,
                "tp_atr_mult": tp_m,
                **result,
            }

    if best_trading_params is None:
        best_trading_params = {
            "threshold": 0.15, "sl_atr_mult": 2.0, "tp_atr_mult": 3.5,
            "sharpe": 0, "n_trades": 0, "return_pct": 0, "win_rate": 0,
        }

    if verbose:
        print(f"  Mejor trading: thr={best_trading_params['threshold']}, "
              f"SL={best_trading_params['sl_atr_mult']}xATR, "
              f"TP={best_trading_params['tp_atr_mult']}xATR "
              f"(Sharpe={best_trading_score:.2f})")

    # ── FASE 3: Evaluar en test set (out-of-sample) ──
    if verbose:
        print(f"\n  Fase 3: Evaluacion out-of-sample (test)...")

    y_pred_te = best_pipeline.predict(X_te)
    test_result = evaluate_trading_params(
        y_te, y_pred_te, atr_te,
        best_trading_params["threshold"],
        best_trading_params["sl_atr_mult"],
        best_trading_params["tp_atr_mult"],
        SPREAD_COST_PCT,
    )

    mask_nz_te = np.abs(y_te) > 0.01
    dir_te = np.mean(np.sign(y_pred_te[mask_nz_te]) == np.sign(y_te[mask_nz_te])) * 100 if mask_nz_te.sum() > 0 else 50

    if verbose:
        print(f"  Test: Dir={dir_te:.1f}%  Sharpe={test_result['sharpe']:.2f}  "
              f"Trades={test_result['n_trades']}  WinR={test_result['win_rate']:.1f}%")

    return {
        "model_params": best_model_params,
        "trading_params": {
            "threshold": best_trading_params["threshold"],
            "sl_atr_mult": best_trading_params["sl_atr_mult"],
            "tp_atr_mult": best_trading_params["tp_atr_mult"],
        },
        "validation": {
            "sharpe": best_trading_score,
            "model_score": round(float(best_model_score), 1),
        },
        "test": {
            "directional_accuracy": round(float(dir_te), 2),
            **test_result,
        },
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  OPTIMIZACION DE PARAMETROS v3.0 - XAUUSD ONLY")
    print("=" * 65)
    print(f"\n  Grid del modelo: {len(list(itertools.product(*MODEL_GRID.values())))} combinaciones")
    print(f"  Grid de trading: {len(list(itertools.product(*TRADING_GRID.values())))} combinaciones")

    archivo = f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n  ERROR: {archivo} no encontrado")
        print(f"  -> Ejecuta primero: python 2_descargar_datos.py")
        exit(1)

    print(f"\n{'='*65}")
    print(f"  OPTIMIZANDO: XAUUSD")
    print(f"{'='*65}")

    df = pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[df["high"] >= df["low"]]

    print(f"  Datos: {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

    result = optimize_xauusd(df)

    # ── RESUMEN ──
    print(f"\n\n{'='*65}")
    print(f"  PARAMETROS OPTIMOS - XAUUSD")
    print(f"{'='*65}")

    mp = result["model_params"]
    tp = result["trading_params"]
    te = result["test"]
    print(f"\n  Modelo: trees={mp['n_estimators']}, depth={mp['max_depth']}, leaf={mp['min_samples_leaf']}")
    print(f"  Trading: thr={tp['threshold']}, SL={tp['sl_atr_mult']}xATR, TP={tp['tp_atr_mult']}xATR")
    print(f"  Test: Dir={te['directional_accuracy']:.1f}%, Sharpe={te['sharpe']:.2f}, "
          f"WinR={te['win_rate']:.1f}%, Trades={te['n_trades']}")

    # Guardar
    opt_path = f"{CARPETA_MODELO}/optimal_params.json"
    with open(opt_path, "w") as f:
        json.dump({"XAUUSD": result}, f, indent=2, ensure_ascii=False)

    # Actualizar config/symbols.json con parámetros óptimos
    os.makedirs("config", exist_ok=True)
    sym_cfg = {
        "XAUUSD": {
            "threshold_buy": tp["threshold"],
            "threshold_sell": -tp["threshold"],
            "sl_atr_mult": tp["sl_atr_mult"],
            "tp_atr_mult": tp["tp_atr_mult"],
            "max_lots": 5.0,
            "max_drawdown_pct": 20.0,
            "vol_regime_min": 0.6,
            "vol_regime_max": 1.8,
        }
    }
    sym_path = "config/symbols.json"
    with open(sym_path, "w") as f:
        json.dump(sym_cfg, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {opt_path}")
    print(f"  Config por simbolo: {sym_path}")
    print("=" * 65)
