# HybridAI Trading System

Sistema de trading algorítmico (Expert Advisor) para MetaTrader 5 que utiliza un modelo de Machine Learning (ExtraTrees) convertido a ONNX. Opera buscando retornos a futuro en símbolos como EURUSD, XAUUSD, GBPUSD y USDJPY.

## Estructura
- `1_verificar_instalacion.py`: Valida si MT5 y Python están integrados correctamente.
- `2_descargar_datos.py`: Se conecta con MT5 para procesar hasta 4 años de barras históricas en timeframe M15 de manera optimizada y automática.
- `3_entrenar_modelo.py`: Lee todo el Dataset, computa variables técnicas avanzadas e indicadores para crear un modelo de Árboles de Regresión aleatorios. Lo exporta como archivo .onnx.
- `HybridAI_EA.mq5`: Algoritmo base para incorporar al MetaEditor y simular/ejecutar órdenes comerciales basadas en el resultado de la IA.

## Uso
1. Instalar requerimientos y revisar instalación (Paso 1 y 2).
2. Descargar los datos teniendo MT5 abierto en una cuenta.
3. Entrenar el modelo IA con el script en Python.
4. Mover el archivo ONNX resultante a la carpeta local `MT5 > MQL5/Files`.
5. Compilar el EA.

**Nota:** Toda ejecución conlleva un riesgo. No apto para cuentas reales sin extrema verificación técnica y backtests de control.
