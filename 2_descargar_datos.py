# =============================================================
#  PASO 2 - DESCARGAR DATOS HISTÓRICOS MULTI-TIMEFRAME v4.0
#  Descarga XAUUSD en M15, H1, H4 y D1.
#  El modelo necesita contexto multi-TF para detectar
#  cambios de estructura y operar en ambas direcciones.
#
#  Comando: python 2_descargar_datos.py
#
#  REQUISITO: MetaTrader 5 debe estar abierto y conectado.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO_BUSCAR = ["XAUUSD", "GOLD", "XAUUSD.", "XAUUSDm"]
TIMEFRAMES = {
    "m15": mt5.TIMEFRAME_M15,
    "h1":  mt5.TIMEFRAME_H1,
    "h4":  mt5.TIMEFRAME_H4,
    "d1":  mt5.TIMEFRAME_D1,
}
ANOS_HISTORIAL = 5
CARPETA_DATOS  = "datos"
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_DATOS, exist_ok=True)

print("=" * 60)
print("  DESCARGA MULTI-TIMEFRAME v4.0 - XAUUSD")
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

# ── Buscar símbolo XAUUSD ────────────────────────────────────
simbolo_encontrado = None
for variante in SIMBOLO_BUSCAR:
    info = mt5.symbol_info(variante)
    if info is not None:
        simbolo_encontrado = variante
        break

if simbolo_encontrado is None:
    print(f"  ERROR: XAUUSD no encontrado. Variantes: {SIMBOLO_BUSCAR}")
    mt5.shutdown()
    exit(1)

if not mt5.symbol_info(simbolo_encontrado).visible:
    mt5.symbol_select(simbolo_encontrado, True)

print(f"  Simbolo: {simbolo_encontrado}")

# ── Descargar cada timeframe ─────────────────────────────────
for tf_name, tf_enum in TIMEFRAMES.items():
    print(f"\n{'-'*50}")
    print(f"  Descargando {tf_name.upper()}...")

    rates = mt5.copy_rates_from_pos(simbolo_encontrado, tf_enum, 0, 120000)

    if rates is None or len(rates) == 0:
        print(f"  ERROR: No se obtuvieron datos {tf_name}: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )

    n_raw = len(df)

    # Limpieza
    n_dups = df.index.duplicated().sum()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["open"] > 0)]
    df = df[df["high"] >= df["low"]]

    returns = df["close"].pct_change().abs()
    n_outliers = (returns > 0.10).sum()
    df = df[returns <= 0.10]

    n_cleaned = n_raw - len(df)

    archivo = f"{CARPETA_DATOS}/xauusd_{tf_name}.csv"
    df.to_csv(archivo)

    print(f"  {len(df):,} barras -> {archivo}")
    print(f"  Inicio: {df.index[0].strftime('%d/%m/%Y %H:%M')}")
    print(f"  Fin:    {df.index[-1].strftime('%d/%m/%Y %H:%M')}")
    print(f"  Limpieza: {n_cleaned} filas removidas")

mt5.shutdown()

print(f"\n{'='*60}")
print(f"  DESCARGA COMPLETADA - 4 TIMEFRAMES")
print(f"\n  Archivos:")
for tf_name in TIMEFRAMES:
    archivo = f"{CARPETA_DATOS}/xauusd_{tf_name}.csv"
    if os.path.exists(archivo):
        print(f"    {archivo}")
print(f"\n  -> Continua con: python 3_entrenar_modelo.py")
print("=" * 60)
