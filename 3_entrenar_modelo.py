# =============================================================
#  PASO 3 - ENTRENAR MODELO IA v4.0 - MULTI-TIMEFRAME
#
#  CAMBIOS CLAVE vs v3.0:
#  1. Features multi-TF: H1, H4, D1 alineados con cada barra M15
#  2. Features de estructura: higher-highs, lower-lows, swing
#  3. Target balanceado: clasificación 3 clases (BUY/SELL/NEUTRAL)
#     en vez de regresión pura (elimina sesgo direccional)
#  4. El modelo aprende CUÁNDO comprar Y cuándo vender
#
#  Comando: python 3_entrenar_modelo.py
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
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"
BARRAS_FUTURO   = 5      # Horizonte de predicción en M15
ATR_THRESHOLD   = 1.0    # Señal si movimiento > 1.0 ATR
OPSET_ONNX      = 12
TRAIN_RATIO     = 0.80

# Modelo con regularización moderada
MODEL_PARAMS = {
    "n_estimators":     600,
    "max_depth":        12,
    "min_samples_leaf": 40,
    "max_features":     "sqrt",
    "class_weight":     "balanced",   # CLAVE: balancea BUY/SELL/NEUTRAL
    "n_jobs":           -1,
    "random_state":     42,
}
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_MODELO, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  HELPERS                                                  ║
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
# ║  FEATURES M15 BASE (20 features: índices 0-19)           ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_features_m15(df):
    """20 features técnicos base en M15."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    feats = np.full((n, 20), np.nan, dtype=np.float64)

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

    return feats, atr14


# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURES MULTI-TIMEFRAME (features 20-39)               ║
# ║  Cada TF aporta: RSI, MACD_hist, EMA_trend, ATR_ratio,   ║
# ║                  ret_reciente                             ║
# ║  H1:  features 20-24                                     ║
# ║  H4:  features 25-29                                     ║
# ║  D1:  features 30-34                                     ║
# ║  Estructura: features 35-39                               ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_features_htf(df_m15, df_h1, df_h4, df_d1):
    """Calcula 20 features adicionales de timeframes superiores + estructura."""
    n = len(df_m15)
    feats = np.full((n, 20), np.nan, dtype=np.float64)

    m15_times = df_m15.index

    # Para cada HTF: alinear al M15 usando merge_asof o búsqueda
    for tf_idx, (df_htf, tf_name, col_offset) in enumerate([
        (df_h1, "H1", 0), (df_h4, "H4", 5), (df_d1, "D1", 10)
    ]):
        c_htf = df_htf["close"].values.astype(np.float64)
        h_htf = df_htf["high"].values.astype(np.float64)
        l_htf = df_htf["low"].values.astype(np.float64)
        n_htf = len(c_htf)

        # Calcular indicadores en el HTF
        rsi_p = 14
        delta = np.diff(c_htf)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = np.full(n_htf, np.nan); al = np.full(n_htf, np.nan)
        if n_htf > rsi_p:
            ag[rsi_p] = np.mean(gain[:rsi_p]); al[rsi_p] = np.mean(loss[:rsi_p])
            for i in range(rsi_p + 1, n_htf):
                ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
                al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
        rs = np.where(al == 0, 100.0, ag / al)
        rsi_htf = 100.0 - (100.0 / (1.0 + rs)); rsi_htf[:rsi_p] = np.nan

        ema12 = ema_calc(c_htf, 12); ema26 = ema_calc(c_htf, 26)
        macd_htf = ema12 - ema26
        macd_sig_htf = ema_calc(macd_htf, 9)
        macd_hist_htf = macd_htf - macd_sig_htf

        ema21_htf = ema_calc(c_htf, 21); ema50_htf = ema_calc(c_htf, 50)
        atr_htf = atr_calc(h_htf, l_htf, c_htf, 14)
        atr_s_htf = np.where(atr_htf == 0, 1e-10, atr_htf)

        # Retorno reciente (3 barras del HTF)
        ret3_htf = np.full(n_htf, np.nan)
        ret3_htf[3:] = (c_htf[3:] - c_htf[:-3]) / c_htf[:-3] * 100.0

        # Alinear: para cada barra M15, encontrar la última barra HTF cerrada
        htf_times = df_htf.index
        htf_idx_map = np.searchsorted(htf_times, m15_times, side='right') - 1

        for i in range(n):
            j = htf_idx_map[i]
            if j < 0 or j >= n_htf:
                continue

            # Feature 0: RSI del HTF normalizado
            feats[i, col_offset + 0] = rsi_htf[j] / 100.0 if not np.isnan(rsi_htf[j]) else np.nan

            # Feature 1: MACD histograma normalizado por ATR
            if not np.isnan(macd_hist_htf[j]) and atr_s_htf[j] > 0:
                feats[i, col_offset + 1] = macd_hist_htf[j] / atr_s_htf[j]

            # Feature 2: Tendencia EMA = (EMA21 - EMA50) / ATR
            if not np.isnan(ema21_htf[j]) and not np.isnan(ema50_htf[j]) and atr_s_htf[j] > 0:
                feats[i, col_offset + 2] = (ema21_htf[j] - ema50_htf[j]) / atr_s_htf[j]

            # Feature 3: ATR ratio (volatilidad relativa del HTF)
            if not np.isnan(atr_htf[j]) and c_htf[j] > 0:
                feats[i, col_offset + 3] = atr_htf[j] / c_htf[j]

            # Feature 4: Retorno reciente del HTF
            if not np.isnan(ret3_htf[j]):
                feats[i, col_offset + 4] = ret3_htf[j]

    # ── Features de ESTRUCTURA (35-39) ──
    c_m15 = df_m15["close"].values.astype(np.float64)
    h_m15 = df_m15["high"].values.astype(np.float64)
    l_m15 = df_m15["low"].values.astype(np.float64)

    # 35: Posición relativa en rango de 96 barras (1 día M15)
    #     (close - min96) / (max96 - min96) → 0=fondo, 1=techo
    for i in range(96, n):
        max96 = np.max(h_m15[i-96:i])
        min96 = np.min(l_m15[i-96:i])
        rng96 = max96 - min96
        feats[i, 15] = (c_m15[i] - min96) / rng96 if rng96 > 0 else 0.5

    # 36: Higher-highs / Lower-lows ratio (20 barras)
    #     Cuenta si los últimos 20 highs forman HH vs LH
    for i in range(20, n):
        highs_window = h_m15[i-20:i]
        hh_count = 0
        for k in range(1, len(highs_window)):
            if highs_window[k] > highs_window[k-1]:
                hh_count += 1
        feats[i, 16] = hh_count / 19.0  # normalizado [0,1]

    # 37: Distancia al máximo de 200 barras (normalizado por ATR)
    atr_m15 = atr_calc(h_m15, l_m15, c_m15, 14)
    atr_m15_s = np.where(atr_m15 == 0, 1e-10, atr_m15)
    for i in range(200, n):
        max200 = np.max(h_m15[i-200:i])
        feats[i, 17] = (max200 - c_m15[i]) / atr_m15_s[i]

    # 38: Distancia al mínimo de 200 barras (normalizado por ATR)
    for i in range(200, n):
        min200 = np.min(l_m15[i-200:i])
        feats[i, 18] = (c_m15[i] - min200) / atr_m15_s[i]

    # 39: Hora cíclica (sin, cos codificados como single feature)
    hours = m15_times.hour + m15_times.minute / 60.0
    feats[:, 19] = np.sin(2.0 * np.pi * hours / 24.0)

    return feats


# ╔══════════════════════════════════════════════════════════╗
# ║  TARGET: CLASIFICACIÓN 3 CLASES                          ║
# ║  0 = NEUTRAL (movimiento < ATR_THRESHOLD * ATR)          ║
# ║  1 = BUY     (sube > ATR_THRESHOLD * ATR)                ║
# ║  2 = SELL    (baja > ATR_THRESHOLD * ATR)                ║
# ╚══════════════════════════════════════════════════════════╝

def crear_target(close, atr, barras_futuro, atr_threshold):
    """Crea target de 3 clases basado en movimiento relativo al ATR."""
    n = len(close)
    y = np.full(n, -1, dtype=np.int32)  # -1 = sin label

    for i in range(n - barras_futuro):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        future_ret = close[i + barras_futuro] - close[i]
        threshold = atr[i] * atr_threshold

        if future_ret > threshold:
            y[i] = 1   # BUY
        elif future_ret < -threshold:
            y[i] = 2   # SELL
        else:
            y[i] = 0   # NEUTRAL

    return y


N_FEATURES = 40  # 20 M15 + 5 H1 + 5 H4 + 5 D1 + 5 estructura

# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  ENTRENAMIENTO v4.0 - MULTI-TIMEFRAME + CLASIFICACIÓN")
    print("=" * 60)

    # ── Cargar datos ──
    archivos = {
        "m15": f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv",
        "h1":  f"{CARPETA_DATOS}/{SIMBOLO}_h1.csv",
        "h4":  f"{CARPETA_DATOS}/{SIMBOLO}_h4.csv",
        "d1":  f"{CARPETA_DATOS}/{SIMBOLO}_d1.csv",
    }

    for tf, path in archivos.items():
        if not os.path.exists(path):
            print(f"\n  ERROR: {path} no encontrado")
            print(f"  -> Ejecuta primero: python 2_descargar_datos.py")
            exit(1)

    dfs = {}
    for tf, path in archivos.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[df["high"] >= df["low"]]
        dfs[tf] = df
        print(f"  {tf.upper()}: {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

    df_m15 = dfs["m15"]

    # ── Calcular features ──
    print(f"\n  Calculando features M15 (20)...")
    X_m15, atr_m15 = calcular_features_m15(df_m15)

    print(f"  Calculando features HTF (15 + 5 estructura)...")
    X_htf = calcular_features_htf(df_m15, dfs["h1"], dfs["h4"], dfs["d1"])

    X_full = np.hstack([X_m15, X_htf])
    print(f"  Total features: {X_full.shape[1]}")

    # ── Target ──
    c = df_m15["close"].values
    y = crear_target(c, atr_m15, BARRAS_FUTURO, ATR_THRESHOLD)

    # Filtrar muestras válidas
    valido = (y >= 0) & ~np.any(np.isnan(X_full), axis=1)
    X_v = X_full[valido].astype(np.float32)
    y_v = y[valido]

    print(f"\n  Muestras validas: {len(X_v):,}")
    print(f"  Distribución de clases:")
    for cls, name in [(0, "NEUTRAL"), (1, "BUY"), (2, "SELL")]:
        cnt = np.sum(y_v == cls)
        print(f"    {name}: {cnt:,} ({cnt/len(y_v)*100:.1f}%)")

    # ── Split temporal ──
    idx = int(len(X_v) * TRAIN_RATIO)
    X_train, y_train = X_v[:idx], y_v[:idx]
    X_test, y_test = X_v[idx:], y_v[idx:]

    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Shuffle train (mantener integridad temporal en test)
    rng_np = np.random.default_rng(42)
    shuffle_idx = rng_np.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # ── ENTRENAR ──
    print(f"\n  Entrenando ExtraTrees Clasificador v4.0...")
    print(f"    class_weight=balanced (fuerza balance BUY/SELL)")
    print(f"    max_depth={MODEL_PARAMS['max_depth']}, "
          f"min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesClassifier(**MODEL_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    print("  Entrenamiento completado")

    # ── EVALUACIÓN ──
    y_pred_tr = pipeline.predict(X_train)
    y_pred_te = pipeline.predict(X_test)

    acc_tr = np.mean(y_pred_tr == y_train) * 100
    acc_te = np.mean(y_pred_te == y_test) * 100

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  Accuracy train: {acc_tr:.1f}%")
    print(f"  Accuracy test:  {acc_te:.1f}%")
    print(f"  Gap: {acc_tr - acc_te:.1f}pp")

    print(f"\n  Classification Report (TEST):")
    target_names = ["NEUTRAL", "BUY", "SELL"]
    print(classification_report(y_test, y_pred_te, target_names=target_names))

    # Verificar balance de predicciones
    print(f"  Distribución de predicciones (TEST):")
    for cls, name in [(0, "NEUTRAL"), (1, "BUY"), (2, "SELL")]:
        cnt = np.sum(y_pred_te == cls)
        print(f"    {name}: {cnt:,} ({cnt/len(y_pred_te)*100:.1f}%)")

    # Verificar que no hay sesgo excesivo
    buy_preds = np.sum(y_pred_te == 1)
    sell_preds = np.sum(y_pred_te == 2)
    if buy_preds > 0 and sell_preds > 0:
        ratio = buy_preds / sell_preds
        print(f"\n  Ratio BUY/SELL predicciones: {ratio:.2f}")
        if ratio > 2.0:
            print(f"  [ALERTA: Sesgo BUY excesivo]")
        elif ratio < 0.5:
            print(f"  [ALERTA: Sesgo SELL excesivo]")
        else:
            print(f"  [OK - predicciones balanceadas]")

    # ── Feature importance ──
    print(f"\n{'='*60}")
    print(f"  IMPORTANCIA DE FEATURES (Top 15)")
    print(f"{'='*60}")
    feature_names = [
        # M15 base (0-19)
        "m15_rsi", "m15_macd", "m15_macd_sig", "m15_macd_hist",
        "m15_atr_pct", "m15_bb_pctb", "m15_bb_width",
        "m15_ema9_dist", "m15_ema21_dist", "m15_ema50_dist",
        "m15_ret1", "m15_ret3", "m15_ret5", "m15_ret10", "m15_ret20",
        "m15_vol_ratio", "m15_hl_ratio", "m15_close_pos", "m15_body_ratio", "m15_willr",
        # H1 (20-24)
        "h1_rsi", "h1_macd_hist", "h1_ema_trend", "h1_atr_pct", "h1_ret3",
        # H4 (25-29)
        "h4_rsi", "h4_macd_hist", "h4_ema_trend", "h4_atr_pct", "h4_ret3",
        # D1 (30-34)
        "d1_rsi", "d1_macd_hist", "d1_ema_trend", "d1_atr_pct", "d1_ret3",
        # Estructura (35-39)
        "struct_pos_range96", "struct_hh_ratio", "struct_dist_max200",
        "struct_dist_min200", "struct_hour_sin",
    ]
    importances = pipeline.named_steps['model'].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, fi in enumerate(sorted_idx[:15]):
        name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
        print(f"  {rank+1:>2}. {name:<22} {importances[fi]:.4f}")

    # ── EXPORTAR ONNX ──
    print(f"\n  Exportando modelo ONNX...")
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]

    try:
        onnx_model = convert_sklearn(
            pipeline, initial_types=initial_type, target_opset=OPSET_ONNX,
            options={id(pipeline.named_steps['model']): {'zipmap': False}},
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
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  ONNX outputs: {output_names}")

    muestra = X_test[:100].astype(np.float32)
    pred_sk = pipeline.predict(muestra)

    # Para clasificadores, ONNX produce [label, probabilities]
    onnx_out = sess.run(None, {input_name: muestra})
    pred_on = onnx_out[0].flatten() if len(onnx_out) > 0 else np.array([])

    match = np.mean(pred_on == pred_sk) * 100 if len(pred_on) > 0 else 0
    print(f"  Validacion ONNX: {match:.1f}% match", end="")
    print("  [OK]" if match > 99 else f"  [ALERTA: {match:.1f}%]")

    # ── GUARDAR CONFIG ──
    config = {
        "version": "4.0",
        "tipo_modelo": "clasificacion_3_clases",
        "clases": {"0": "NEUTRAL", "1": "BUY", "2": "SELL"},
        "activo": "XAUUSD",
        "n_features": N_FEATURES,
        "barras_futuro": BARRAS_FUTURO,
        "atr_threshold": ATR_THRESHOLD,
        "accuracy_test": round(float(acc_te), 2),
        "onnx_input_name": input_name,
        "onnx_output_names": output_names,
        "model_params": {k: str(v) for k, v in MODEL_PARAMS.items()},
        "feature_names": feature_names,
        "timeframes_usados": ["M15", "H1", "H4", "D1"],
    }

    ruta_cfg = f"{CARPETA_MODELO}/modelo_config.json"
    with open(ruta_cfg, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── RESUMEN ──
    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO v4.0 COMPLETADO")
    print(f"{'='*60}")
    print(f"  Tipo: Clasificación 3 clases (BUY/SELL/NEUTRAL)")
    print(f"  Features: {N_FEATURES} ({20} M15 + 15 HTF + 5 estructura)")
    print(f"  Accuracy test: {acc_te:.1f}%")
    print(f"  Balance: class_weight=balanced")
    print(f"\n  Archivos generados:")
    print(f"    1. {ruta_onnx}")
    print(f"    2. {ruta_cfg}")
    print(f"\n  Proximo paso:")
    print(f"    python 4_backtesting.py")
    print("=" * 60)
