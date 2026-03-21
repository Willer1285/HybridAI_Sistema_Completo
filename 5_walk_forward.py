# =============================================================
#  PASO 5 - VALIDACIÓN WALK-FORWARD
#  Simula el despliegue real del modelo: entrena en ventanas
#  históricas y evalúa en periodos futuros nunca vistos.
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
SIMBOLOS       = ["xauusd", "eurusd", "gbpusd", "usdjpy"]
SYMBOL_IDS     = {s: i for i, s in enumerate(SIMBOLOS)}
CARPETA_DATOS  = "datos"
CARPETA_MODELO = "modelo"
N_FEATURES_BASE = 20
N_FEATURES      = 24
BARRAS_FUTURO   = 5

# Walk-Forward params
TRAIN_BARS    = 4 * 24192 // 5   # ~3 años de M15 por símbolo
VAL_BARS      = 24192 // 4       # ~3 meses
TEST_BARS     = 24192 // 4       # ~3 meses
STEP_BARS     = 24192 // 4       # Avanza 3 meses

# Modelo
MODEL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 15,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE FEATURES (mismas que backtesting)          ║
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

def calcular_features_array(c, h, l, o, v):
    n = len(c)
    feats = np.full((n, N_FEATURES_BASE), np.nan, dtype=np.float64)

    rsi_p = 14
    delta = np.diff(c)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n, np.nan); al = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p]); al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs  = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:rsi_p] = np.nan

    ema12 = ema_calc(c, 12); ema26 = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema_calc(macd_line, 9)
    macd_hist = macd_line - macd_sig

    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    sma20 = sma_calc(c, 20)
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19:i+1], ddof=0)
    bb_up = sma20 + 2.0 * std20; bb_lo = sma20 - 2.0 * std20
    bb_w  = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp  = np.where(sma20 == 0, np.nan, bb_w / sma20)

    ema9  = ema_calc(c, 9); ema21 = ema_calc(c, 21); ema50 = ema_calc(c, 50)

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

    feats[:, 0]  = rsi / 100.0
    feats[:, 1]  = macd_line / atr_s
    feats[:, 2]  = macd_sig  / atr_s
    feats[:, 3]  = macd_hist / atr_s
    feats[:, 4]  = np.where(c == 0, np.nan, atr14 / c)
    feats[:, 5]  = bb_pctb
    feats[:, 6]  = bb_bwp
    feats[:, 7]  = np.where(ema9  == 0, np.nan, (c - ema9)  / ema9  * 100.0)
    feats[:, 8]  = np.where(ema21 == 0, np.nan, (c - ema21) / ema21 * 100.0)
    feats[:, 9]  = np.where(ema50 == 0, np.nan, (c - ema50) / ema50 * 100.0)
    feats[:, 10] = ret(c, 1)
    feats[:, 11] = ret(c, 3)
    feats[:, 12] = ret(c, 5)
    feats[:, 13] = ret(c, 10)
    feats[:, 14] = ret(c, 20)
    feats[:, 15] = vol_rt
    feats[:, 16] = hl_r
    feats[:, 17] = cl_p
    feats[:, 18] = bd_r
    feats[:, 19] = will

    return feats, atr14


def agregar_symbol_id(X_base, symbol_name, n_symbols=4):
    n = X_base.shape[0]
    sym_id = SYMBOL_IDS.get(symbol_name, 0)
    one_hot = np.zeros((n, n_symbols), dtype=np.float64)
    one_hot[:, sym_id] = 1.0
    return np.hstack([X_base, one_hot])


# ╔══════════════════════════════════════════════════════════╗
# ║  WALK-FORWARD VALIDATION                                 ║
# ╚══════════════════════════════════════════════════════════╝

def walk_forward_symbol(df, simbolo, verbose=True):
    """Ejecuta walk-forward validation para un símbolo."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    X_base, atr_vals = calcular_features_array(c, h, l, o, v)
    X_full = agregar_symbol_id(X_base, simbolo)

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
            print(f"  Insuficientes datos para walk-forward en {simbolo.upper()}")
        return []

    # Obtener índices válidos
    valid_indices = np.where(valid_mask)[0]
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

        # Simular retornos de trading simple
        buy_mask = y_pred > 0.10
        sell_mask = y_pred < -0.10
        buy_ret = y_te[buy_mask]
        sell_ret = -y_te[sell_mask]
        trade_rets = np.concatenate([buy_ret, sell_ret]) if (len(buy_ret) + len(sell_ret)) > 0 else np.array([0])
        n_trades = len(trade_rets)
        mean_ret = np.mean(trade_rets) if n_trades > 0 else 0
        std_ret = np.std(trade_rets) if n_trades > 1 else 1
        sharpe_approx = mean_ret / std_ret * np.sqrt(24192) if std_ret > 0 else 0

        fold_result = {
            "fold": fold,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "rmse": round(float(rmse), 5),
            "directional_accuracy": round(float(dir_acc), 2),
            "n_trades": n_trades,
            "mean_trade_return": round(float(mean_ret), 4),
            "sharpe_approx": round(float(sharpe_approx), 2),
            "profitable": mean_ret > 0,
        }
        results.append(fold_result)

        if verbose:
            status = "OK" if mean_ret > 0 else "NEG"
            print(f"    Fold {fold:>2}: Dir={dir_acc:5.1f}%  RMSE={rmse:.4f}  "
                  f"Trades={n_trades:>4}  Ret={mean_ret:+.3f}%  [{status}]")

        start += STEP_BARS

    return results


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  WALK-FORWARD VALIDATION - HybridAI v2.0")
    print("=" * 65)
    print(f"\n  Ventana train:  ~{TRAIN_BARS:,} barras (~3 anios)")
    print(f"  Ventana test:   ~{TEST_BARS:,} barras (~3 meses)")
    print(f"  Paso de avance: ~{STEP_BARS:,} barras (~3 meses)")

    all_wf_results = {}

    for simbolo in SIMBOLOS:
        archivo = f"{CARPETA_DATOS}/{simbolo}_m15.csv"
        if not os.path.exists(archivo):
            print(f"\n  {archivo} no encontrado, saltando...")
            continue

        print(f"\n{'='*65}")
        print(f"  WALK-FORWARD: {simbolo.upper()}")
        print(f"{'='*65}")

        df = pd.read_csv(archivo, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[df["high"] >= df["low"]]

        print(f"  Datos: {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

        results = walk_forward_symbol(df, simbolo)

        if results:
            n_folds = len(results)
            n_profitable = sum(1 for r in results if r["profitable"])
            avg_dir = np.mean([r["directional_accuracy"] for r in results])
            avg_sharpe = np.mean([r["sharpe_approx"] for r in results])
            consistency = n_profitable / n_folds * 100

            print(f"\n  --- Resumen {simbolo.upper()} ---")
            print(f"  Folds totales:       {n_folds}")
            print(f"  Folds rentables:     {n_profitable}/{n_folds} ({consistency:.0f}%)")
            print(f"  Precision media:     {avg_dir:.1f}%")
            print(f"  Sharpe medio:        {avg_sharpe:.2f}")

            all_wf_results[simbolo.upper()] = {
                "folds": results,
                "n_folds": n_folds,
                "n_profitable": n_profitable,
                "consistency_pct": round(consistency, 1),
                "avg_directional_accuracy": round(float(avg_dir), 2),
                "avg_sharpe": round(float(avg_sharpe), 2),
            }
        else:
            all_wf_results[simbolo.upper()] = {"error": "Datos insuficientes"}

    # ── RESUMEN FINAL ──
    print(f"\n\n{'='*65}")
    print(f"  RESUMEN WALK-FORWARD - TODOS LOS SIMBOLOS")
    print(f"{'='*65}")
    print(f"  {'Simbolo':<10} {'Folds':>6} {'Rentables':>10} {'Consist%':>9} {'AvgDir%':>8} {'AvgSharpe':>10}")
    print(f"  {'-'*58}")

    for sym, data in all_wf_results.items():
        if "error" in data:
            print(f"  {sym:<10} {'---':>6} {'ERROR':>10}")
            continue
        print(f"  {sym:<10} {data['n_folds']:>6} "
              f"{data['n_profitable']}/{data['n_folds']:>8} "
              f"{data['consistency_pct']:>9.0f} "
              f"{data['avg_directional_accuracy']:>8.1f} "
              f"{data['avg_sharpe']:>10.2f}")

    # Guardar resultados
    wf_path = f"{CARPETA_MODELO}/walk_forward_results.json"
    with open(wf_path, "w") as f:
        json.dump(all_wf_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {wf_path}")
    print(f"\n  Proximo paso:")
    print(f"    python 6_optimizar_parametros.py  (optimizacion de hiperparametros)")
    print("=" * 65)
