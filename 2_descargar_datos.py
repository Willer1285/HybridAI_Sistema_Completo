# =============================================================
#  PASO 2 - DESCARGAR DATOS HISTÓRICOS DESDE MT5
#  Descarga 5 años de datos M15 para los 4 símbolos.
#  Comando: python 2_descargar_datos.py
#
#  REQUISITO: MetaTrader 5 debe estar abierto y conectado
#             al broker antes de ejecutar este script.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLOS_BUSCAR = {
    "XAUUSD": ["XAUUSD", "GOLD", "XAUUSD."],
    "EURUSD": ["EURUSD", "EURUSD."],
    "GBPUSD": ["GBPUSD", "GBPUSD."],
    "USDJPY": ["USDJPY", "USDJPY."],
}
TIMEFRAME    = mt5.TIMEFRAME_M15
ANOS_HISTORIAL = 5
CARPETA_DATOS  = "datos"
# ─────────────────────────────────────────────────────────────

os.makedirs(CARPETA_DATOS, exist_ok=True)

print("=" * 55)
print("  DESCARGA DE DATOS HISTÓRICOS - MT5")
print("=" * 55)

# ── Conectar a MT5 ───────────────────────────────────────────
if not mt5.initialize():
    print(f"\n❌ ERROR: No se pudo conectar a MT5")
    print(f"   → Abre MetaTrader 5 primero")
    print(f"   → Error: {mt5.last_error()}")
    exit(1)

print(f"\n✅ MT5 conectado - Build {mt5.version()[1]}")
print(f"   Broker: {mt5.terminal_info().company}")

fecha_inicio = datetime.now() - timedelta(days=365 * ANOS_HISTORIAL)
fecha_fin    = datetime.now()

print(f"\n   Período: {fecha_inicio.strftime('%d/%m/%Y')} → {fecha_fin.strftime('%d/%m/%Y')}")
print(f"   Timeframe: M15")

simbolos_ok = []

for nombre_base, variantes in SIMBOLOS_BUSCAR.items():
    print(f"\n{'─'*45}")
    print(f"  Descargando {nombre_base}...")

    simbolo_encontrado = None
    for variante in variantes:
        info = mt5.symbol_info(variante)
        if info is not None:
            simbolo_encontrado = variante
            break

    if simbolo_encontrado is None:
        print(f"  ❌ {nombre_base} no encontrado en tu broker.")
        print(f"     Variantes probadas: {variantes}")
        print(f"     → Revisa el nombre exacto en MT5 y agrega al script")
        continue

    # Activar símbolo si no está visible
    if not mt5.symbol_info(simbolo_encontrado).visible:
        mt5.symbol_select(simbolo_encontrado, True)

    # 5 años de datos M15 = aprox 250 dias/año * 24 horas * 4 barras = 24,000 por año -> 120,000 barras
    rates = mt5.copy_rates_from_pos(
        simbolo_encontrado, TIMEFRAME, 0, 120000
    )

    if rates is None or len(rates) == 0:
        print(f"  ❌ No se obtuvieron datos: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )

    # Eliminar filas con precios inválidos
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

    archivo = f"{CARPETA_DATOS}/{nombre_base.lower()}_m15.csv"
    df.to_csv(archivo)

    print(f"  ✅ {len(df):,} barras guardadas → {archivo}")
    print(f"     Inicio: {df.index[0].strftime('%d/%m/%Y %H:%M')}")
    print(f"     Fin:    {df.index[-1].strftime('%d/%m/%Y %H:%M')}")
    print(f"     Precio actual: {df['close'].iloc[-1]:.5f}")

    simbolos_ok.append(nombre_base)

mt5.shutdown()

print(f"\n{'='*55}")
if simbolos_ok:
    print(f"  ✅ DESCARGA COMPLETADA")
    print(f"     Símbolos descargados: {simbolos_ok}")
    print(f"\n  → Continúa con: python 3_entrenar_modelo.py")
else:
    print(f"  ❌ No se descargó ningún símbolo")
    print(f"     Verifica los nombres exactos en MT5")
print("=" * 55)
