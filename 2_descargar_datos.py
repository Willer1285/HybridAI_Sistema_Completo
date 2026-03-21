# =============================================================
#  PASO 2 - DESCARGAR DATOS HISTÓRICOS DESDE MT5 (OPTIMIZADO)
#  Descarga 5 años de datos M15 para los 4 símbolos.
#  v2.0: Limpieza mejorada (duplicados, gaps, outliers)
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
SIMBOLOS_BUSCAR = {
    "XAUUSD": ["XAUUSD", "GOLD", "XAUUSD.", "XAUUSDm"],
    "EURUSD": ["EURUSD", "EURUSD.", "EURUSDm"],
    "GBPUSD": ["GBPUSD", "GBPUSD.", "GBPUSDm"],
    "USDJPY": ["USDJPY", "USDJPY.", "USDJPYm"],
}
TIMEFRAME    = mt5.TIMEFRAME_M15
ANOS_HISTORIAL = 5
CARPETA_DATOS  = "datos"
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_DATOS, exist_ok=True)

print("=" * 60)
print("  DESCARGA DE DATOS HISTORICOS - MT5 v2.0")
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

simbolos_ok = []

for nombre_base, variantes in SIMBOLOS_BUSCAR.items():
    print(f"\n{'-'*50}")
    print(f"  Descargando {nombre_base}...")

    simbolo_encontrado = None
    for variante in variantes:
        info = mt5.symbol_info(variante)
        if info is not None:
            simbolo_encontrado = variante
            break

    if simbolo_encontrado is None:
        print(f"  {nombre_base} no encontrado en tu broker.")
        print(f"  Variantes probadas: {variantes}")
        print(f"  -> Revisa el nombre exacto en MT5 y agrega al script")
        continue

    # Activar símbolo si no está visible
    if not mt5.symbol_info(simbolo_encontrado).visible:
        mt5.symbol_select(simbolo_encontrado, True)

    # Descargar datos
    rates = mt5.copy_rates_from_pos(
        simbolo_encontrado, TIMEFRAME, 0, 120000
    )

    if rates is None or len(rates) == 0:
        print(f"  No se obtuvieron datos: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )

    n_raw = len(df)

    # ── LIMPIEZA MEJORADA v2.0 ──

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
    # Reinsertar primera fila si fue eliminada
    if len(df) > 0 and df.index[0] != df.index[0]:
        pass  # pct_change produce NaN en primera fila, se mantiene

    # 6. Verificar y reportar gaps temporales
    time_diffs = df.index.to_series().diff()
    expected_gap = pd.Timedelta(minutes=15)
    # Gaps mayores a 3 horas (excluir fines de semana normales)
    significant_gaps = time_diffs[time_diffs > pd.Timedelta(hours=3)]
    n_gaps = len(significant_gaps)

    n_cleaned = n_raw - len(df)

    archivo = f"{CARPETA_DATOS}/{nombre_base.lower()}_m15.csv"
    df.to_csv(archivo)

    print(f"  {len(df):,} barras guardadas -> {archivo}")
    print(f"  Inicio: {df.index[0].strftime('%d/%m/%Y %H:%M')}")
    print(f"  Fin:    {df.index[-1].strftime('%d/%m/%Y %H:%M')}")
    print(f"  Precio actual: {df['close'].iloc[-1]:.5f}")
    print(f"  Limpieza: {n_cleaned} filas removidas ({n_dups} dups, {n_outliers} outliers)")
    if n_gaps > 0:
        print(f"  Gaps >3h detectados: {n_gaps} (normales si incluyen fines de semana)")

    simbolos_ok.append(nombre_base)

mt5.shutdown()

print(f"\n{'='*60}")
if simbolos_ok:
    print(f"  DESCARGA COMPLETADA")
    print(f"  Simbolos descargados: {simbolos_ok}")
    print(f"\n  -> Continua con: python 3_entrenar_modelo.py")
else:
    print(f"  No se descargo ningun simbolo")
    print(f"  Verifica los nombres exactos en MT5")
print("=" * 60)
