# =============================================================
#  PASO 1 - VERIFICAR INSTALACIÓN
#  Ejecuta este script para confirmar que todo está instalado
#  correctamente antes de continuar.
#  Comando: python 1_verificar_instalacion.py
# =============================================================

import sys

print("=" * 55)
print("  VERIFICACIÓN DEL SISTEMA HybridAI Trading")
print("=" * 55)
print(f"\nPython: {sys.version}")

errores = []

# ── Verificar librerías ──────────────────────────────────────
libs = {
    "MetaTrader5"  : "MetaTrader5",
    "pandas"       : "pandas",
    "numpy"        : "numpy",
    "scikit-learn" : "sklearn",
    "skl2onnx"     : "skl2onnx",
    "onnx"         : "onnx",
    "onnxruntime"  : "onnxruntime",
}

print("\n--- Librerías ---")
for nombre, modulo in libs.items():
    try:
        m = __import__(modulo)
        version = getattr(m, "__version__", "OK")
        print(f"  ✅ {nombre:<15} v{version}")
    except ImportError:
        print(f"  ❌ {nombre:<15} NO INSTALADA")
        errores.append(nombre)

# ── Verificar conexión MT5 ───────────────────────────────────
print("\n--- Conexión MetaTrader 5 ---")
try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        info = mt5.terminal_info()
        print(f"  ✅ MT5 conectado")
        print(f"     Build:    {mt5.version()[1]}")
        print(f"     Broker:   {info.company}")
        print(f"     Programa: {info.name}")
        account = mt5.account_info()
        if account:
            print(f"     Servidor: {account.server}")
        print(f"     Ruta:     {info.path}")
        mt5.shutdown()
    else:
        print(f"  ❌ MT5 no conectado - Error: {mt5.last_error()}")
        print("     → Abre MetaTrader 5 y vuelve a ejecutar este script")
        errores.append("conexion_MT5")
except Exception as e:
    print(f"  ❌ Error al conectar MT5: {e}")
    errores.append("conexion_MT5")

# ── Resultado final ──────────────────────────────────────────
print("\n" + "=" * 55)
if not errores:
    print("  ✅ TODO LISTO. Puedes continuar con el Paso 2.")
else:
    print("  ❌ HAY ERRORES. Corrígelos antes de continuar:")
    for e in errores:
        if e == "conexion_MT5":
            print(f"     → Abre MetaTrader 5 primero")
        else:
            print(f"     → pip install {e}")
print("=" * 55)
