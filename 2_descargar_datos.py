# =============================================================
#  PASO 2 - DESCARGAR DATOS HISTÓRICOS DESDE MT5 v3.0
#  Descarga 5 años de datos M15 solo para XAUUSD.
#
#  Comando: python 2_descargar_datos.py
#
#  REQUISITO: MetaTrader 5 debe estar abierto y conectado
#             al broker antes de ejecutar este script.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO_BUSCAR = ["XAUUSD", "GOLD", "XAUUSD.", "XAUUSDm"]
TIMEFRAME      = mt5.TIMEFRAME_M15
ANOS_HISTORIAL = 5
CARPETA_DATOS  = "datos"
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_DATOS, exist_ok=True)

print("=" * 60)
print("  DESCARGA DE DATOS HISTORICOS v3.0 - XAUUSD ONLY")
print("=" * 60)

# ── Conectar a MT5 ───────────────────────────────────────────
if not mt5.initialize():
    print(f"\n  ERROR: No se pudo conectar a MT5")
    print(f"  -> Abre MetaTrader 5 primero")
    print(f"  -> Error: {mt5.last_error()}")
    exit(1)

print(f"\n  MT5 conectado - Build {mt5.version()[1]}")
print(f"  Broker: {mt5.terminal_info().company}")

fecha_inicio = datetime.now() - timedelta(days=365 * ANOS_HISTORIAL)
fecha_fin    = datetime.now()

print(f"\n  Periodo: {fecha_inicio.strftime('%d/%m/%Y')} -> {fecha_fin.strftime('%d/%m/%Y')}")
print(f"  Timeframe: M15")

# ── Buscar símbolo XAUUSD ────────────────────────────────────
print(f"\n{'-'*50}")
print(f"  Buscando XAUUSD...")

simbolo_encontrado = None
for variante in SIMBOLO_BUSCAR:
    info = mt5.symbol_info(variante)
    if info is not None:
        simbolo_encontrado = variante
        break

if simbolo_encontrado is None:
    print(f"  ERROR: XAUUSD no encontrado en tu broker.")
    print(f"  Variantes probadas: {SIMBOLO_BUSCAR}")
    print(f"  -> Revisa el nombre exacto en MT5 y agrega al script")
    mt5.shutdown()
    exit(1)

# Activar símbolo si no está visible
if not mt5.symbol_info(simbolo_encontrado).visible:
    mt5.symbol_select(simbolo_encontrado, True)

# Descargar datos
print(f"  Simbolo encontrado: {simbolo_encontrado}")
print(f"  Descargando datos...")

rates = mt5.copy_rates_from_pos(
    simbolo_encontrado, TIMEFRAME, 0, 120000
)

if rates is None or len(rates) == 0:
    print(f"  ERROR: No se obtuvieron datos: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df.set_index("time", inplace=True)
df = df[["open", "high", "low", "close", "tick_volume"]].rename(
    columns={"tick_volume": "volume"}
)

n_raw = len(df)

# ── LIMPIEZA ──

# 1. Eliminar duplicados de timestamp
n_dups = df.index.duplicated().sum()
df = df[~df.index.duplicated(keep='first')]

# 2. Ordenar cronológicamente
df = df.sort_index()

# 3. Eliminar precios inválidos
df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open"] > 0)]

# 4. Eliminar filas donde high < low (datos corruptos)
df = df[df["high"] >= df["low"]]

# 5. Eliminar outliers extremos (cambios > 10% en una barra M15)
returns = df["close"].pct_change().abs()
n_outliers = (returns > 0.10).sum()
df = df[returns <= 0.10]

# 6. Verificar y reportar gaps temporales
time_diffs = df.index.to_series().diff()
significant_gaps = time_diffs[time_diffs > pd.Timedelta(hours=3)]
n_gaps = len(significant_gaps)

n_cleaned = n_raw - len(df)

archivo = f"{CARPETA_DATOS}/xauusd_m15.csv"
df.to_csv(archivo)

mt5.shutdown()

print(f"\n  {len(df):,} barras guardadas -> {archivo}")
print(f"  Inicio: {df.index[0].strftime('%d/%m/%Y %H:%M')}")
print(f"  Fin:    {df.index[-1].strftime('%d/%m/%Y %H:%M')}")
print(f"  Precio actual: {df['close'].iloc[-1]:.5f}")
print(f"  Limpieza: {n_cleaned} filas removidas ({n_dups} dups, {n_outliers} outliers)")
if n_gaps > 0:
    print(f"  Gaps >3h detectados: {n_gaps} (normales si incluyen fines de semana)")

print(f"\n{'='*60}")
print(f"  DESCARGA COMPLETADA - XAUUSD")
print(f"\n  -> Continua con: python 3_entrenar_modelo.py")
print("=" * 60)
