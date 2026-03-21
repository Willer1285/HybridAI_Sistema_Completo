# =============================================================
#  PASO 3 - ENTRENAR MODELO IA Y EXPORTAR A ONNX
#  Entrena un modelo ExtraTrees universal para los 4 símbolos
#  y lo exporta al formato ONNX para usarlo en el EA de MT5.
#
#  Comando: python 3_entrenar_modelo.py
#
#  Tiempo estimado: 3-8 minutos dependiendo de tu PC
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
CARPETA_DATOS  = "datos"
CARPETA_MODELO = "modelo"
N_FEATURES     = 20
BARRAS_FUTURO  = 5      # Predice el retorno en las próximas 5 barras M15 (~75 min)
OPSET_ONNX     = 12
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_MODELO, exist_ok=True)

# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIÓN: CALCULAR 20 FEATURES TÉCNICOS                 ║
# ║  ⚠️  ESTA LÓGICA DEBE SER IDÉNTICA AL EA MQL5           ║
# ╚══════════════════════════════════════════════════════════╝
def calcular_features(df):
    """
    Calcula los 20 indicadores técnicos que alimentan el modelo.
    
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

    feats = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

    # ─────────── helpers internos ─────────────────────────────

    def ema_calc(data, period):
        result = np.full(len(data), np.nan)
        
        first_valid = 0
        while first_valid < len(data) and np.isnan(data[first_valid]):
            first_valid += 1
            
        if first_valid + period > len(data):
            return result
            
        k = 2.0 / (period + 1)
        # Primer valor es un SMA
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
    atr_s = np.where(atr14 == 0, 1e-10, atr14)  # safe divide

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


# ╔══════════════════════════════════════════════════════════╗
# ║  CARGAR Y COMBINAR DATOS DE LOS 4 SÍMBOLOS              ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 55)
print("  ENTRENAMIENTO DEL SISTEMA HybridAI")
print("=" * 55)

all_X, all_y = [], []
simbolos_cargados = []

for simbolo in SIMBOLOS:
    archivo = f"{CARPETA_DATOS}/{simbolo}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n⚠️  {archivo} no encontrado, saltando...")
        continue

    print(f"\n📊 Cargando {simbolo.upper()}...")
    df = pd.read_csv(archivo, index_col=0, parse_dates=True)
    print(f"   {len(df):,} barras  |  {df.index[0].date()} → {df.index[-1].date()}")

    X = calcular_features(df)
    c = df["close"].values

    # Target: retorno % a las próximas BARRAS_FUTURO barras
    y = np.full(len(c), np.nan)
    y[:-BARRAS_FUTURO] = (
        (c[BARRAS_FUTURO:] - c[:-BARRAS_FUTURO]) / c[:-BARRAS_FUTURO] * 100.0
    )

    # Filtrar NaN/Inf
    valido = ~(np.any(np.isnan(X), axis=1) | np.isnan(y) | np.isinf(y))
    X_v = X[valido]
    y_v = np.clip(y[valido], -10.0, 10.0)

    print(f"   Muestras válidas: {len(X_v):,}")
    all_X.append(X_v)
    all_y.append(y_v)
    simbolos_cargados.append(simbolo.upper())

if not all_X:
    print("\n❌ ERROR: No se encontraron archivos de datos.")
    print("   → Ejecuta primero: python 2_descargar_datos.py")
    exit(1)

X_total = np.vstack(all_X).astype(np.float32)
y_total = np.concatenate(all_y).astype(np.float32)

print(f"\n✅ Total muestras combinadas: {len(X_total):,}")
print(f"   Símbolos: {simbolos_cargados}")

# ── Split temporal 80/20 ─────────────────────────────────────
idx     = int(len(X_total) * 0.80)
X_train = X_total[:idx];  y_train = y_total[:idx]
X_test  = X_total[idx:];  y_test  = y_total[idx:]
print(f"\n   Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ╔══════════════════════════════════════════════════════════╗
# ║  ENTRENAR MODELO (StandardScaler + ExtraTrees)          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n🤖 Entrenando modelo ExtraTreesRegressor...")
print("   (puede tardar 3-8 minutos, por favor espera...)")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ExtraTreesRegressor(
        n_estimators   = 500,
        max_depth      = 12,
        min_samples_leaf = 10,
        max_features   = "sqrt",
        n_jobs         = -1,   # usa todos los núcleos del CPU
        random_state   = 42,
    )),
])

pipeline.fit(X_train, y_train)
print("   ✅ Entrenamiento completado")


# ╔══════════════════════════════════════════════════════════╗
# ║  EVALUACIÓN                                              ║
# ╚══════════════════════════════════════════════════════════╝
y_pred_tr = pipeline.predict(X_train)
y_pred_te = pipeline.predict(X_test)

rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_te))
mae_te  = mean_absolute_error(y_test, y_pred_te)

# Precisión direccional (lo más importante para trading)
dir_tr = np.mean(np.sign(y_pred_tr) == np.sign(y_train)) * 100
dir_te = np.mean(np.sign(y_pred_te) == np.sign(y_test))  * 100

print(f"\n📊 RESULTADOS DEL MODELO:")
print(f"   {'Métrica':<35} {'Train':>8}  {'Test':>8}")
print(f"   {'─'*55}")
print(f"   {'RMSE (error cuadrático medio %)':<35} {rmse_tr:>8.4f}  {rmse_te:>8.4f}")
print(f"   {'MAE  (error medio absoluto %)':<35} {'─':>8}  {mae_te:>8.4f}")
print(f"   {'Precisión direccional (%)':<35} {dir_tr:>8.1f}  {dir_te:>8.1f}")
print(f"\n   ℹ️  Precisión >52% es mejor que operar al azar")

if dir_te < 51:
    print(f"\n   ⚠️  Precisión baja. El sistema funcionará pero con señales conservadoras.")
    print(f"      Los filtros de umbral del EA protegerán el capital.")


# ╔══════════════════════════════════════════════════════════╗
# ║  EXPORTAR A ONNX                                         ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n📦 Exportando modelo a formato ONNX...")

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
    print(f"   ✅ ONNX guardado: {ruta_onnx}")
    print(f"   Tamaño: {tam_kb:.0f} KB")

except Exception as e:
    print(f"   ❌ Error al exportar ONNX: {e}")
    print(f"   Intenta: pip install --upgrade skl2onnx onnx")
    exit(1)


# ╔══════════════════════════════════════════════════════════╗
# ║  VALIDAR QUE EL ONNX FUNCIONA CORRECTAMENTE             ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n🔍 Validando modelo ONNX...")
sess        = rt.InferenceSession(ruta_onnx)
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape
out_shape   = sess.get_outputs()[0].shape

print(f"   Input:  '{input_name}' {input_shape}")
print(f"   Output: '{output_name}' {out_shape}")

# Comparar predicciones sklearn vs ONNX
muestra = X_test[:10].astype(np.float32)
pred_sk = pipeline.predict(muestra)
pred_on = sess.run([output_name], {input_name: muestra})[0].flatten()

diff_max = float(np.max(np.abs(pred_on - pred_sk)))
print(f"   Diferencia máxima sklearn↔ONNX: {diff_max:.6f}")

if diff_max < 0.01:
    print(f"   ✅ ONNX validado correctamente (diferencia < 0.01)")
else:
    print(f"   ⚠️  Diferencia mayor a la esperada ({diff_max:.4f})")
    print(f"      El modelo funcionará pero puede haber pequeñas diferencias")


# ╔══════════════════════════════════════════════════════════╗
# ║  GUARDAR CONFIGURACIÓN PARA EL EA                        ║
# ╚══════════════════════════════════════════════════════════╝
config = {
    "version"               : "1.0",
    "n_features"            : N_FEATURES,
    "barras_futuro"         : BARRAS_FUTURO,
    "simbolos_entrenados"   : simbolos_cargados,
    "precision_test_pct"    : round(dir_te, 2),
    "rmse_test"             : round(float(rmse_te), 5),
    "onnx_input_name"       : input_name,
    "onnx_output_name"      : output_name,
    "umbral_compra"         : 0.25,
    "umbral_venta"          : -0.25,
    "feature_names": [
        "rsi14_norm", "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "atr_pct", "bb_pctb", "bb_width_pct",
        "ema9_dist_pct", "ema21_dist_pct", "ema50_dist_pct",
        "ret1", "ret3", "ret5", "ret10", "ret20",
        "vol_ratio", "hl_ratio", "close_pos", "body_ratio", "willr_norm",
    ],
    "instrucciones_mt5": (
        "Copia modelo/hybrid_ai_model.onnx a la carpeta "
        "MQL5\\Files\\ de tu instalación de MetaTrader 5"
    )
}

ruta_cfg = f"{CARPETA_MODELO}/modelo_config.json"
with open(ruta_cfg, "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n✅ Configuración guardada: {ruta_cfg}")

# ── RESUMEN FINAL ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ✅ ENTRENAMIENTO Y EXPORTACIÓN COMPLETADOS")
print(f"{'='*55}")
print(f"\n  🎯 Precisión direccional (test): {dir_te:.1f}%")
print(f"\n  📁 ARCHIVOS GENERADOS:")
print(f"     1. {ruta_onnx}")
print(f"        → Este es el modelo que carga el EA en MT5")
print(f"     2. {ruta_cfg}")
print(f"        → Configuración de referencia")
print(f"\n  📋 PRÓXIMO PASO:")
print(f"     Copia el archivo ONNX a la carpeta de MT5:")
print(f"     C:\\Users\\TuUsuario\\AppData\\Roaming\\MetaQuotes\\")
print(f"              Terminal\\<ID_TERMINAL>\\MQL5\\Files\\")
print(f"\n  Luego abre MetaEditor y compila HybridAI_EA.mq5")
print("=" * 55)
