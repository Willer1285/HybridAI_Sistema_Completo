# HybridAI Trading System v2.0

Sistema de trading algoritmico (Expert Advisor) para MetaTrader 5 que utiliza un modelo de Machine Learning (ExtraTrees) convertido a ONNX. Opera buscando retornos a futuro en 4 activos: XAUUSD, EURUSD, GBPUSD y USDJPY.

## Cambios v2.0

- **Split temporal corregido**: Cada simbolo se divide 80/20 individualmente antes de combinar, eliminando contaminacion entre activos.
- **Symbol ID**: 4 features one-hot (24 total) permiten al modelo aprender patrones especificos por activo.
- **Evaluacion por simbolo**: Metricas individuales para cada activo (precision direccional, RMSE, MAE).
- **Umbrales optimizados**: Busqueda automatica del umbral optimo por simbolo.
- **Motor de backtesting Python**: Simulacion de trading con costos reales (spread, comisiones, slippage).
- **Walk-forward validation**: Validacion temporal deslizante que simula despliegue real.
- **Optimizacion de parametros**: Grid search de hiperparametros del modelo + parametros de trading.
- **EA mejorado**: Trailing stop, filtro de spread, deteccion de filling mode, cierre de contrarias.
- **Limpieza de datos**: Deteccion de duplicados, outliers, gaps temporales.

## Estructura

| Archivo | Descripcion |
|---------|-------------|
| `1_verificar_instalacion.py` | Valida si MT5 y Python estan integrados correctamente |
| `2_descargar_datos.py` | Descarga y limpia 5 anios de datos M15 desde MT5 |
| `3_entrenar_modelo.py` | Entrena modelo con split correcto, symbol_id, y evaluacion por activo |
| `4_backtesting.py` | Motor de backtesting con metricas financieras completas |
| `5_walk_forward.py` | Validacion walk-forward (simula despliegue real) |
| `6_optimizar_parametros.py` | Optimizacion de hiperparametros y parametros de trading |
| `HybridAI_EA.mq5` | Expert Advisor v2.0 (24 features, trailing stop, spread filter) |
| `config/symbols.json` | Configuracion especifica por activo |

## Flujo de uso

```
1. python 1_verificar_instalacion.py    # Verificar entorno
2. python 2_descargar_datos.py          # Descargar datos (MT5 abierto)
3. python 3_entrenar_modelo.py          # Entrenar y exportar ONNX
4. python 4_backtesting.py              # Evaluar rendimiento de trading
5. python 5_walk_forward.py             # Validar estabilidad temporal
6. python 6_optimizar_parametros.py     # Optimizar parametros
7. Copiar modelo ONNX a MT5/MQL5/Files/ y compilar EA
```

## Metricas de evaluacion

El sistema ahora calcula por cada activo:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (% y duracion)
- Profit Factor, Win Rate, Expectancia
- Precision direccional
- Trades por mes

## Nota

Toda ejecucion conlleva un riesgo. No apto para cuentas reales sin extrema verificacion tecnica y backtests de control.
