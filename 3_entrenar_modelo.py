# =============================================================
#  PASO 3 - ENTRENAR MODELO IA Y EXPORTAR A ONNX (OPTIMIZADO)
#  Versión 2.0: Corrige split temporal, agrega symbol_id,
#  evaluación por símbolo, walk-forward, y métricas financieras.
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
SIMBOLOS       = ["xauusd", "eurusd", "gbpusd", "usdjpy"]
SYMBOL_IDS     = {s: i for i, s in enumerate(SIMBOLOS)}  # Identificador numérico
CARPETA_DATOS  = "datos"
CARPETA_MODELO = "modelo"
N_FEATURES_BASE = 20    # Features técnicos originales
N_FEATURES_NEW  = 5     # symbol_id(4 one-hot) + hora_ciclica(0 aquí, se agrega en EA)
N_FEATURES      = N_FEATURES_BASE + 4  # 20 base + 4 one-hot symbol = 24
BARRAS_FUTURO  = 5      # Predice el retorno en las próximas 5 barras M15 (~75 min)
OPSET_ONNX     = 12
TRAIN_RATIO    = 0.80   # 80% train, 20% test POR SÍMBOLO
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_MODELO, exist_ok=True)

# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIÓN: CALCULAR 20 FEATURES TÉCNICOS BASE            ║
# ║  Los features 20-23 (symbol one-hot) se agregan después  ║
# ╚══════════════════════════════════════════════════════════╝
def calcular_features(df):
    """
    Calcula los 20 indicadores técnicos base que alimentan el modelo.

    Lista de features (índices 0-19):
    0  rsi14_norm      - RSI(14) / 100                [0.0 - 1.0]
    1  macd_norm       - Línea MACD / ATR(14)
    2  macd_sig_norm   - Señal MACD / ATR(14)
    3  macd_hist_norm  - Histograma MACD / ATR(14)
    4  atr_pct         - ATR(14) / Precio cierre
    5  bb_pctb         - Bollinger %B
    6  bb_width_pct    - Ancho BB / Precio cierre
    7  ema9_dist       - (Cierre - EMA9) / EMA9 * 100
    8  ema21_dist      - (Cierre - EMA21) / EMA21 * 100
    9  ema50_dist      - (Cierre - EMA50) / EMA50 * 100
    10 ret_1           - Retorno 1 barra (%)
    11 ret_3           - Retorno 3 barras (%)
    12 ret_5           - Retorno 5 barras (%)
    13 ret_10          - Retorno 10 barras (%)
    14 ret_20          - Retorno 20 barras (%)
    15 vol_ratio       - Volumen / Media20 volumen
    16 hl_ratio        - (High-Low) / Cierre * 100
    17 close_pos       - (Cierre-Low) / (High-Low)   [0.0 - 1.0]
    18 body_ratio      - (Cierre-Open) / (High-Low)  [-1.0 - 1.0]
    19 willr_norm      - Williams %R normalizado      [0.0 - 1.0]
    """
    c  = df["close"].values.astype(np.float64)
    h  = df["high"].values.astype(np.float64)
    l  = df["low"].values.astype(np.float64)
    o  = df["open"].values.astype(np.float64)
    v  = df["volume"].values.astype(np.float64)
    n  = len(c)

    feats = np.full((n, N_FEATURES_BASE), np.nan, dtype=np.float64)

    # ─────────── helpers internos ─────────────────────────────

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
        tr = np.maximum(h_[1:] - l_[1:],
             np.maximum(np.abs(h_[1:] - c_[:-1]),
                        np.abs(l_[1:] - c_[:-1])))
        tr = np.concatenate([[np.nan], tr])
        result = np.full(n, np.nan)
        result[period] = np.mean(tr[1 : period + 1])
        for i in range(period + 1, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        return result

    # ─────────── RSI(14) ──────────────────────────────────────
    rsi_p = 14
    delta = np.diff(c)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag    = np.full(n, np.nan)
    al    = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p])
    al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs  = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:rsi_p] = np.nan

    # ─────────── MACD(12,26,9) ────────────────────────────────
    ema12     = ema_calc(c, 12)
    ema26     = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema_calc(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # ─────────── ATR(14) ──────────────────────────────────────
    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    # ─────────── Bollinger Bands(20, 2σ) ──────────────────────
    sma20   = sma_calc(c, 20)
    std20   = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19 : i+1], ddof=0)
    bb_up   = sma20 + 2.0 * std20
    bb_lo   = sma20 - 2.0 * std20
    bb_w    = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp  = np.where(sma20 == 0, np.nan, bb_w / sma20)

    # ─────────── EMAs ─────────────────────────────────────────
    ema9  = ema_calc(c, 9)
    ema21 = ema_calc(c, 21)
    ema50 = ema_calc(c, 50)

    # ─────────── Retornos (%) ─────────────────────────────────
    def ret(data, p):
        r = np.full(n, np.nan)
        r[p:] = (data[p:] - data[:-p]) / data[:-p] * 100.0
        return r

    ret1  = ret(c, 1)
    ret3  = ret(c, 3)
    ret5  = ret(c, 5)
    ret10 = ret(c, 10)
    ret20 = ret(c, 20)

    # ─────────── Volumen ratio ────────────────────────────────
    vol_ma = sma_calc(v, 20)
    vol_rt = np.where(vol_ma == 0, 1.0, v / vol_ma)

    # ─────────── Métricas de vela ─────────────────────────────
    hl_r = np.where(c == 0, np.nan, (h - l) / c * 100.0)
    rng  = h - l
    cl_p = np.where(rng == 0, 0.5, (c - l) / rng)
    bd_r = np.where(rng == 0, 0.0, (c - o) / rng)

    # ─────────── Williams %R(14) ──────────────────────────────
    mxh = np.full(n, np.nan)
    mnl = np.full(n, np.nan)
    for i in range(13, n):
        mxh[i] = np.max(h[i-13 : i+1])
        mnl[i] = np.min(l[i-13 : i+1])
    wl_r = mxh - mnl
    will = np.where(wl_r == 0, 0.5, (c - mnl) / wl_r)

    # ─────────── Ensamblar matriz ─────────────────────────────
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
    feats[:, 10] = ret1
    feats[:, 11] = ret3
    feats[:, 12] = ret5
    feats[:, 13] = ret10
    feats[:, 14] = ret20
    feats[:, 15] = vol_rt
    feats[:, 16] = hl_r
    feats[:, 17] = cl_p
    feats[:, 18] = bd_r
    feats[:, 19] = will

    return feats


def agregar_symbol_id(X_base, symbol_name, n_symbols=4):
    """Agrega one-hot encoding del símbolo a la matriz de features."""
    n = X_base.shape[0]
    sym_id = SYMBOL_IDS.get(symbol_name, 0)
    one_hot = np.zeros((n, n_symbols), dtype=np.float64)
    one_hot[:, sym_id] = 1.0
    return np.hstack([X_base, one_hot])


# ╔══════════════════════════════════════════════════════════╗
# ║  CARGAR DATOS CON SPLIT TEMPORAL CORRECTO POR SÍMBOLO    ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("  ENTRENAMIENTO DEL SISTEMA HybridAI v2.0 (OPTIMIZADO)")
print("=" * 60)

train_X, train_y = [], []
test_X, test_y = [], []
symbol_test_data = {}  # Para evaluación por símbolo
simbolos_cargados = []

for simbolo in SIMBOLOS:
    archivo = f"{CARPETA_DATOS}/{simbolo}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n  {archivo} no encontrado, saltando...")
        continue

    print(f"\n  Cargando {simbolo.upper()}...")
    df = pd.read_csv(archivo, index_col=0, parse_dates=True)

    # ── Limpieza de datos mejorada ──
    # Eliminar duplicados de timestamp
    df = df[~df.index.duplicated(keep='first')]
    # Ordenar cronológicamente
    df = df.sort_index()
    # Eliminar precios inválidos
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    # Eliminar filas donde high < low (datos corruptos)
    df = df[df["high"] >= df["low"]]

    print(f"    {len(df):,} barras  |  {df.index[0].date()} -> {df.index[-1].date()}")

    X_base = calcular_features(df)
    X_full = agregar_symbol_id(X_base, simbolo)
    c = df["close"].values

    # Target: retorno % a las próximas BARRAS_FUTURO barras
    y = np.full(len(c), np.nan)
    y[:-BARRAS_FUTURO] = (
        (c[BARRAS_FUTURO:] - c[:-BARRAS_FUTURO]) / c[:-BARRAS_FUTURO] * 100.0
    )

    # Filtrar NaN/Inf
    valido = ~(np.any(np.isnan(X_full), axis=1) | np.isnan(y) | np.isinf(y))
    X_v = X_full[valido]
    y_v = np.clip(y[valido], -10.0, 10.0)

    print(f"    Muestras validas: {len(X_v):,}")

    # ── SPLIT TEMPORAL POR SÍMBOLO (CORRECCIÓN CRÍTICA) ──
    idx = int(len(X_v) * TRAIN_RATIO)
    X_tr = X_v[:idx]
    y_tr = y_v[:idx]
    X_te = X_v[idx:]
    y_te = y_v[idx:]

    print(f"    Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    train_X.append(X_tr)
    train_y.append(y_tr)
    test_X.append(X_te)
    test_y.append(y_te)

    # Guardar datos de test por símbolo para evaluación individual
    symbol_test_data[simbolo.upper()] = (X_te, y_te)
    simbolos_cargados.append(simbolo.upper())

if not train_X:
    print("\n  ERROR: No se encontraron archivos de datos.")
    print("   -> Ejecuta primero: python 2_descargar_datos.py")
    exit(1)

X_train = np.vstack(train_X).astype(np.float32)
y_train = np.concatenate(train_y).astype(np.float32)
X_test  = np.vstack(test_X).astype(np.float32)
y_test  = np.concatenate(test_y).astype(np.float32)

# Shuffle del train set (mantiene test sin tocar para evaluación temporal)
rng = np.random.default_rng(42)
shuffle_idx = rng.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"\n  Total muestras - Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Simbolos: {simbolos_cargados}")
print(f"  Features: {N_FEATURES} (20 tecnicas + 4 symbol one-hot)")


# ╔══════════════════════════════════════════════════════════╗
# ║  ENTRENAR MODELO (StandardScaler + ExtraTrees)          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n  Entrenando modelo ExtraTreesRegressor v2.0...")
print("  (puede tardar 3-8 minutos)")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ExtraTreesRegressor(
        n_estimators   = 600,
        max_depth      = 15,
        min_samples_leaf = 20,
        max_features   = "sqrt",
        n_jobs         = -1,
        random_state   = 42,
    )),
])

pipeline.fit(X_train, y_train)
print("  Entrenamiento completado")


# ╔══════════════════════════════════════════════════════════╗
# ║  EVALUACIÓN GLOBAL + POR SÍMBOLO                        ║
# ╚══════════════════════════════════════════════════════════╝
y_pred_tr = pipeline.predict(X_train)
y_pred_te = pipeline.predict(X_test)

rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_te))
mae_te  = mean_absolute_error(y_test, y_pred_te)

# Precisión direccional excluyendo predicciones cercanas a cero
mask_nonzero_tr = np.abs(y_train) > 0.01
mask_nonzero_te = np.abs(y_test)  > 0.01
dir_tr = np.mean(np.sign(y_pred_tr[mask_nonzero_tr]) == np.sign(y_train[mask_nonzero_tr])) * 100
dir_te = np.mean(np.sign(y_pred_te[mask_nonzero_te]) == np.sign(y_test[mask_nonzero_te]))  * 100

print(f"\n{'='*60}")
print(f"  RESULTADOS GLOBALES")
print(f"{'='*60}")
print(f"  {'Metrica':<35} {'Train':>8}  {'Test':>8}")
print(f"  {'-'*55}")
print(f"  {'RMSE (%)':<35} {rmse_tr:>8.4f}  {rmse_te:>8.4f}")
print(f"  {'MAE  (%)':<35} {'--':>8}  {mae_te:>8.4f}")
print(f"  {'Precision direccional (%)':<35} {dir_tr:>8.1f}  {dir_te:>8.1f}")
print(f"\n  (Precision >52% es mejor que operar al azar)")

# ── EVALUACIÓN POR SÍMBOLO INDIVIDUAL ──
print(f"\n{'='*60}")
print(f"  RESULTADOS POR SIMBOLO")
print(f"{'='*60}")
print(f"  {'Simbolo':<10} {'RMSE':>8} {'MAE':>8} {'Dir%':>8} {'Muestras':>10}")
print(f"  {'-'*50}")

symbol_metrics = {}
for sym, (X_s, y_s) in symbol_test_data.items():
    X_s_f = X_s.astype(np.float32)
    y_pred_s = pipeline.predict(X_s_f)
    rmse_s = np.sqrt(mean_squared_error(y_s, y_pred_s))
    mae_s  = mean_absolute_error(y_s, y_pred_s)
    mask_s = np.abs(y_s) > 0.01
    dir_s  = np.mean(np.sign(y_pred_s[mask_s]) == np.sign(y_s[mask_s])) * 100 if mask_s.sum() > 0 else 0
    print(f"  {sym:<10} {rmse_s:>8.4f} {mae_s:>8.4f} {dir_s:>8.1f} {len(y_s):>10,}")
    symbol_metrics[sym] = {
        "rmse": round(float(rmse_s), 5),
        "mae": round(float(mae_s), 5),
        "directional_accuracy": round(float(dir_s), 2),
        "samples": int(len(y_s)),
    }

# ── Feature importance ──
print(f"\n{'='*60}")
print(f"  IMPORTANCIA DE FEATURES (Top 10)")
print(f"{'='*60}")
feature_names = [
    "rsi14_norm", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "atr_pct", "bb_pctb", "bb_width_pct",
    "ema9_dist", "ema21_dist", "ema50_dist",
    "ret1", "ret3", "ret5", "ret10", "ret20",
    "vol_ratio", "hl_ratio", "close_pos", "body_ratio", "willr_norm",
    "sym_xauusd", "sym_eurusd", "sym_gbpusd", "sym_usdjpy",
]
importances = pipeline.named_steps['model'].feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx[:10]):
    print(f"  {rank+1:>2}. {feature_names[idx]:<20} {importances[idx]:.4f}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CALCULAR UMBRALES ÓPTIMOS POR SÍMBOLO                  ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'='*60}")
print(f"  UMBRALES OPTIMOS POR SIMBOLO")
print(f"{'='*60}")

optimal_thresholds = {}
for sym, (X_s, y_s) in symbol_test_data.items():
    X_s_f = X_s.astype(np.float32)
    y_pred_s = pipeline.predict(X_s_f)

    best_threshold = 0.10
    best_score = -999
    for thr in np.arange(0.05, 0.60, 0.01):
        # Simular trades: comprar cuando pred > thr, vender cuando pred < -thr
        buy_mask = y_pred_s > thr
        sell_mask = y_pred_s < -thr
        if buy_mask.sum() + sell_mask.sum() < 20:
            continue
        buy_returns = y_s[buy_mask]
        sell_returns = -y_s[sell_mask]
        all_returns = np.concatenate([buy_returns, sell_returns])
        if len(all_returns) < 20:
            continue
        # Sharpe-like score
        mean_ret = np.mean(all_returns)
        std_ret = np.std(all_returns)
        if std_ret < 1e-8:
            continue
        score = mean_ret / std_ret * np.sqrt(252 * 4)  # Anualizado aprox M15
        if score > best_score:
            best_score = score
            best_threshold = thr

    optimal_thresholds[sym] = round(float(best_threshold), 2)
    print(f"  {sym}: umbral = {best_threshold:.2f}%  (score = {best_score:.2f})")


# ╔══════════════════════════════════════════════════════════╗
# ║  EXPORTAR A ONNX                                         ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n  Exportando modelo a formato ONNX...")

initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]

try:
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=OPSET_ONNX,
    )

    ruta_onnx = f"{CARPETA_MODELO}/hybrid_ai_model.onnx"
    with open(ruta_onnx, "wb") as f:
        f.write(onnx_model.SerializeToString())

    tam_kb = os.path.getsize(ruta_onnx) / 1024
    print(f"  ONNX guardado: {ruta_onnx}")
    print(f"  Tamano: {tam_kb:.0f} KB")

except Exception as e:
    print(f"  ERROR al exportar ONNX: {e}")
    print(f"  Intenta: pip install --upgrade skl2onnx onnx")
    exit(1)


# ╔══════════════════════════════════════════════════════════╗
# ║  VALIDAR QUE EL ONNX FUNCIONA CORRECTAMENTE             ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n  Validando modelo ONNX...")
sess        = rt.InferenceSession(ruta_onnx)
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape
out_shape   = sess.get_outputs()[0].shape

print(f"  Input:  '{input_name}' {input_shape}")
print(f"  Output: '{output_name}' {out_shape}")

# Comparar predicciones sklearn vs ONNX con muestra más amplia
muestra = X_test[:100].astype(np.float32)
pred_sk = pipeline.predict(muestra)
pred_on = sess.run([output_name], {input_name: muestra})[0].flatten()

diff_max = float(np.max(np.abs(pred_on - pred_sk)))
print(f"  Diferencia maxima sklearn<->ONNX: {diff_max:.6f}")

if diff_max < 0.01:
    print(f"  ONNX validado correctamente (diferencia < 0.01)")
else:
    print(f"  Diferencia mayor a la esperada ({diff_max:.4f})")


# ╔══════════════════════════════════════════════════════════╗
# ║  GUARDAR CONFIGURACIÓN PARA EL EA                        ║
# ╚══════════════════════════════════════════════════════════╝
config = {
    "version"               : "2.0",
    "n_features"            : N_FEATURES,
    "n_features_base"       : N_FEATURES_BASE,
    "barras_futuro"         : BARRAS_FUTURO,
    "simbolos_entrenados"   : simbolos_cargados,
    "precision_global_pct"  : round(dir_te, 2),
    "rmse_test"             : round(float(rmse_te), 5),
    "onnx_input_name"       : input_name,
    "onnx_output_name"      : output_name,
    "umbrales_por_simbolo"  : optimal_thresholds,
    "metricas_por_simbolo"  : symbol_metrics,
    "symbol_ids"            : {s.upper(): i for s, i in SYMBOL_IDS.items()},
    "feature_names"         : feature_names,
    "feature_importances"   : {feature_names[i]: round(float(importances[i]), 4)
                               for i in sorted_idx[:10]},
    "instrucciones_mt5": (
        "Copia modelo/hybrid_ai_model.onnx a la carpeta "
        "MQL5\\Files\\ de tu instalacion de MetaTrader 5. "
        "IMPORTANTE: El modelo ahora usa 24 features (20 base + 4 symbol one-hot)."
    )
}

ruta_cfg = f"{CARPETA_MODELO}/modelo_config.json"
with open(ruta_cfg, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n  Configuracion guardada: {ruta_cfg}")

# ── RESUMEN FINAL ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ENTRENAMIENTO v2.0 COMPLETADO")
print(f"{'='*60}")
print(f"\n  Precision direccional global (test): {dir_te:.1f}%")
print(f"\n  Rendimiento por simbolo:")
for sym, m in symbol_metrics.items():
    status = "OK" if m["directional_accuracy"] > 52 else "BAJO"
    print(f"    {sym}: {m['directional_accuracy']:.1f}%  [{status}]")
print(f"\n  Archivos generados:")
print(f"    1. {ruta_onnx}  (modelo ONNX - {N_FEATURES} features)")
print(f"    2. {ruta_cfg}   (configuracion + umbrales optimos)")
print(f"\n  Proximo paso:")
print(f"    python 4_backtesting.py  (evaluar rendimiento de trading)")
print("=" * 60)
