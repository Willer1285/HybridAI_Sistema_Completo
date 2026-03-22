# =============================================================
#  PASO 3 - ENTRENAR MODELO IA Y EXPORTAR A ONNX (v3.0)
#  Versión 3.0: Solo XAUUSD, features de régimen y tiempo,
#  regularización fuerte anti-overfitting.
#
#  Comando: python 3_entrenar_modelo.py
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"
N_FEATURES_BASE = 20    # Features técnicos base
N_FEATURES      = 24    # 20 base + vol_regime + trend_str + hour_sin + hour_cos
BARRAS_FUTURO   = 5     # Predice retorno en 5 barras M15 (~75 min)
OPSET_ONNX      = 12
TRAIN_RATIO     = 0.80

# Modelo con regularización FUERTE para evitar overfitting
MODEL_PARAMS = {
    "n_estimators":    500,
    "max_depth":       10,    # Era 15 → más conservador
    "min_samples_leaf": 50,   # Era 20 → evita memorizar ruido
    "max_features":    "sqrt",
    "n_jobs":          -1,
    "random_state":    42,
}
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_MODELO, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  HELPERS DE CÁLCULO                                      ║
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


# ╔══════════════════════════════════════════════════════════╗
# ║  CALCULAR 24 FEATURES (v3.0)                             ║
# ║  0-19:  Indicadores técnicos base                        ║
# ║  20:    vol_regime  = ATR(14) / SMA(ATR(14), 50)         ║
# ║  21:    trend_str   = (EMA21 - EMA50) / ATR(14)          ║
# ║  22:    hour_sin    = sin(2π * hora / 24)                ║
# ║  23:    hour_cos    = cos(2π * hora / 24)                ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_features_v3(df):
    """Calcula los 24 features v3.0 para XAUUSD."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    feats = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

    # ── RSI(14) ──
    rsi_p = 14
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n, np.nan)
    al = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p])
    al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:rsi_p] = np.nan

    # ── MACD(12,26,9) ──
    ema12 = ema_calc(c, 12)
    ema26 = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig = ema_calc(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # ── ATR(14) ──
    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    # ── Bollinger Bands(20, 2σ) ──
    sma20 = sma_calc(c, 20)
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19:i+1], ddof=0)
    bb_up = sma20 + 2.0 * std20
    bb_lo = sma20 - 2.0 * std20
    bb_w = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp = np.where(sma20 == 0, np.nan, bb_w / sma20)

    # ── EMAs ──
    ema9 = ema_calc(c, 9)
    ema21 = ema_calc(c, 21)
    ema50 = ema_calc(c, 50)

    # ── Retornos (%) ──
    def ret(data, p):
        r = np.full(n, np.nan)
        r[p:] = (data[p:] - data[:-p]) / data[:-p] * 100.0
        return r

    # ── Volumen ratio ──
    vol_ma = sma_calc(v, 20)
    vol_rt = np.where(vol_ma == 0, 1.0, v / vol_ma)

    # ── Métricas de vela ──
    hl_r = np.where(c == 0, np.nan, (h - l) / c * 100.0)
    rng = h - l
    cl_p = np.where(rng == 0, 0.5, (c - l) / rng)
    bd_r = np.where(rng == 0, 0.0, (c - o) / rng)

    # ── Williams %R(14) ──
    mxh = np.full(n, np.nan)
    mnl = np.full(n, np.nan)
    for i in range(13, n):
        mxh[i] = np.max(h[i-13:i+1])
        mnl[i] = np.min(l[i-13:i+1])
    wl_r = mxh - mnl
    will = np.where(wl_r == 0, 0.5, (c - mnl) / wl_r)

    # ── 20 features base (0-19) ──
    feats[:, 0]  = rsi / 100.0
    feats[:, 1]  = macd_line / atr_s
    feats[:, 2]  = macd_sig / atr_s
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

    # ── 4 features NUEVOS v3.0 (20-23) ──

    # 20: Régimen de volatilidad = ATR(14) / SMA(ATR(14), 50)
    #     >1 = volatilidad creciente, <1 = volatilidad decreciente
    atr_sma50 = sma_calc(atr14, 50)
    feats[:, 20] = np.where(
        (atr_sma50 == 0) | np.isnan(atr_sma50), 1.0, atr14 / atr_sma50
    )

    # 21: Fuerza de tendencia = (EMA21 - EMA50) / ATR(14)
    #     Positivo = tendencia alcista, negativo = bajista, magnitud = fuerza
    feats[:, 21] = np.where(atr_s == 0, 0.0, (ema21 - ema50) / atr_s)

    # 22-23: Codificación cíclica de la hora
    hours = df.index.hour + df.index.minute / 60.0
    feats[:, 22] = np.sin(2.0 * np.pi * hours / 24.0)
    feats[:, 23] = np.cos(2.0 * np.pi * hours / 24.0)

    return feats, atr14


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  ENTRENAMIENTO HybridAI v3.0 - XAUUSD ONLY")
    print("=" * 60)

    archivo = f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n  ERROR: {archivo} no encontrado")
        print(f"  -> Ejecuta primero: python 2_descargar_datos.py")
        exit(1)

    print(f"\n  Cargando {SIMBOLO.upper()}...")
    df = pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[df["high"] >= df["low"]]

    print(f"    {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

    # Calcular features v3.0
    X_full, atr_vals = calcular_features_v3(df)
    c = df["close"].values

    # Target: retorno % próximas 5 barras
    y = np.full(len(c), np.nan)
    y[:-BARRAS_FUTURO] = (c[BARRAS_FUTURO:] - c[:-BARRAS_FUTURO]) / c[:-BARRAS_FUTURO] * 100.0

    # Filtrar NaN/Inf
    valido = ~(np.any(np.isnan(X_full), axis=1) | np.isnan(y) | np.isinf(y))
    X_v = X_full[valido]
    y_v = np.clip(y[valido], -10.0, 10.0)

    print(f"    Muestras validas: {len(X_v):,}")

    # Split temporal
    idx = int(len(X_v) * TRAIN_RATIO)
    X_train = X_v[:idx].astype(np.float32)
    y_train = y_v[:idx].astype(np.float32)
    X_test = X_v[idx:].astype(np.float32)
    y_test = y_v[idx:].astype(np.float32)

    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"    Features: {N_FEATURES} (20 base + vol_regime + trend_str + hour_sin + hour_cos)")

    # Shuffle train
    rng_np = np.random.default_rng(42)
    shuffle_idx = rng_np.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # ── ENTRENAR ──
    print(f"\n  Entrenando ExtraTrees v3.0 (regularizacion fuerte)...")
    print(f"    max_depth={MODEL_PARAMS['max_depth']}, "
          f"min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}, "
          f"n_estimators={MODEL_PARAMS['n_estimators']}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(**MODEL_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    print("  Entrenamiento completado")

    # ── EVALUACIÓN ──
    y_pred_tr = pipeline.predict(X_train)
    y_pred_te = pipeline.predict(X_test)

    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
    mae_te = mean_absolute_error(y_test, y_pred_te)

    mask_nz_tr = np.abs(y_train) > 0.01
    mask_nz_te = np.abs(y_test) > 0.01
    dir_tr = np.mean(np.sign(y_pred_tr[mask_nz_tr]) == np.sign(y_train[mask_nz_tr])) * 100
    dir_te = np.mean(np.sign(y_pred_te[mask_nz_te]) == np.sign(y_test[mask_nz_te])) * 100

    print(f"\n{'='*60}")
    print(f"  RESULTADOS - XAUUSD")
    print(f"{'='*60}")
    print(f"  {'Metrica':<30} {'Train':>8}  {'Test':>8}")
    print(f"  {'-'*50}")
    print(f"  {'RMSE (%)':<30} {rmse_tr:>8.4f}  {rmse_te:>8.4f}")
    print(f"  {'MAE  (%)':<30} {'--':>8}  {mae_te:>8.4f}")
    print(f"  {'Precision direccional (%)':<30} {dir_tr:>8.1f}  {dir_te:>8.1f}")

    overfit_gap = dir_tr - dir_te
    print(f"\n  Gap train-test: {overfit_gap:.1f}pp", end="")
    if overfit_gap > 5:
        print("  [ALERTA: posible overfitting]")
    elif overfit_gap > 3:
        print("  [MODERADO]")
    else:
        print("  [OK - buena generalizacion]")

    # ── Feature importance ──
    print(f"\n{'='*60}")
    print(f"  IMPORTANCIA DE FEATURES (Top 12)")
    print(f"{'='*60}")
    feature_names = [
        "rsi14_norm", "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "atr_pct", "bb_pctb", "bb_width_pct",
        "ema9_dist", "ema21_dist", "ema50_dist",
        "ret1", "ret3", "ret5", "ret10", "ret20",
        "vol_ratio", "hl_ratio", "close_pos", "body_ratio", "willr_norm",
        "vol_regime", "trend_strength", "hour_sin", "hour_cos",
    ]
    importances = pipeline.named_steps['model'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, fi in enumerate(sorted_idx[:12]):
        print(f"  {rank+1:>2}. {feature_names[fi]:<20} {importances[fi]:.4f}")

    # ── UMBRAL ÓPTIMO ──
    print(f"\n{'='*60}")
    print(f"  UMBRAL OPTIMO XAUUSD")
    print(f"{'='*60}")

    best_threshold = 0.10
    best_score = -999
    for thr in np.arange(0.02, 0.40, 0.01):
        buy_mask = y_pred_te > thr
        sell_mask = y_pred_te < -thr
        if buy_mask.sum() + sell_mask.sum() < 20:
            continue
        buy_returns = y_test[buy_mask]
        sell_returns = -y_test[sell_mask]
        all_returns = np.concatenate([buy_returns, sell_returns])
        if len(all_returns) < 20:
            continue
        mean_ret = np.mean(all_returns)
        std_ret = np.std(all_returns)
        if std_ret < 1e-8:
            continue
        score = mean_ret / std_ret * np.sqrt(252 * 4)
        if score > best_score:
            best_score = score
            best_threshold = thr

    print(f"  Umbral optimo: {best_threshold:.2f}%  (Sharpe-score: {best_score:.2f})")

    # ── EXPORTAR ONNX ──
    print(f"\n  Exportando modelo ONNX...")
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]

    try:
        onnx_model = convert_sklearn(
            pipeline, initial_types=initial_type, target_opset=OPSET_ONNX,
        )
        ruta_onnx = f"{CARPETA_MODELO}/hybrid_ai_model.onnx"
        with open(ruta_onnx, "wb") as f:
            f.write(onnx_model.SerializeToString())
        tam_kb = os.path.getsize(ruta_onnx) / 1024
        print(f"  ONNX guardado: {ruta_onnx} ({tam_kb:.0f} KB)")
    except Exception as e:
        print(f"  ERROR ONNX: {e}")
        exit(1)

    # Validar ONNX
    sess = rt.InferenceSession(ruta_onnx)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    muestra = X_test[:100].astype(np.float32)
    pred_sk = pipeline.predict(muestra)
    pred_on = sess.run([output_name], {input_name: muestra})[0].flatten()
    diff_max = float(np.max(np.abs(pred_on - pred_sk)))
    print(f"  Validacion ONNX: diff_max = {diff_max:.6f}", end="")
    print("  [OK]" if diff_max < 0.01 else f"  [ALERTA: {diff_max:.4f}]")

    # ── GUARDAR CONFIG ──
    config = {
        "version": "3.0",
        "activo": "XAUUSD",
        "n_features": N_FEATURES,
        "barras_futuro": BARRAS_FUTURO,
        "precision_direccional_test": round(float(dir_te), 2),
        "rmse_test": round(float(rmse_te), 5),
        "umbral_optimo": round(float(best_threshold), 2),
        "onnx_input_name": input_name,
        "onnx_output_name": output_name,
        "model_params": MODEL_PARAMS,
        "feature_names": feature_names,
        "feature_importances": {
            feature_names[i]: round(float(importances[i]), 4)
            for i in sorted_idx[:12]
        },
        "protecciones": {
            "max_drawdown_pct": 20.0,
            "max_lots": 5.0,
            "vol_regime_min": 0.6,
            "vol_regime_max": 1.8,
            "cooling_bars_after_losses": 5,
            "max_consecutive_losses": 3,
        },
    }

    ruta_cfg = f"{CARPETA_MODELO}/modelo_config.json"
    with open(ruta_cfg, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config guardada: {ruta_cfg}")

    # Actualizar config/symbols.json
    os.makedirs("config", exist_ok=True)
    sym_cfg = {
        "XAUUSD": {
            "threshold_buy": round(float(best_threshold), 2),
            "threshold_sell": round(float(-best_threshold), 2),
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 3.5,
            "symbol_id": 0,
            "max_lots": 5.0,
            "max_drawdown_pct": 20.0,
            "vol_regime_min": 0.6,
            "vol_regime_max": 1.8,
        }
    }
    with open("config/symbols.json", "w") as f:
        json.dump(sym_cfg, f, indent=2, ensure_ascii=False)

    # ── RESUMEN ──
    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO v3.0 COMPLETADO - XAUUSD ONLY")
    print(f"{'='*60}")
    status = "OK" if dir_te > 52 else ("MARGINAL" if dir_te > 50.5 else "BAJO")
    print(f"  Precision: {dir_te:.1f}% [{status}]")
    print(f"  Umbral: {best_threshold:.2f}%")
    print(f"  Overfit gap: {overfit_gap:.1f}pp")
    print(f"\n  Archivos generados:")
    print(f"    1. {ruta_onnx}")
    print(f"    2. {ruta_cfg}")
    print(f"    3. config/symbols.json")
    print(f"\n  Proximo paso:")
    print(f"    python 4_backtesting.py")
    print("=" * 60)
