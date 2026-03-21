# HybridAI - Análisis Exhaustivo y Plan de Optimización

## Diagnóstico: ¿Por qué solo funciona con XAUUSD?

El sistema actual entrena un **modelo universal único** sobre los 4 activos combinados,
pero presenta deficiencias estructurales que favorecen desproporcionadamente a XAUUSD
y perjudican el rendimiento en EURUSD, GBPUSD y USDJPY.

---

# PARTE 1: DEFICIENCIAS Y ERRORES ENCONTRADOS

## 1. ERRORES CRÍTICOS (Impacto directo en pérdidas)

### 1.1 BUG: Inconsistencia de umbrales Config vs EA
- **Archivo**: `3_entrenar_modelo.py:385-386` vs `HybridAI_EA.mq5:25-26`
- **Problema**: El config JSON genera umbrales `0.25 / -0.25` pero el EA usa `0.0015 / -0.0015`
- **Impacto**: El EA opera con umbrales ~167x más sensibles que los recomendados, generando señales excesivas y falsas

```python
# modelo_config.json genera:
"umbral_compra": 0.25,    # 0.25%
"umbral_venta": -0.25,    # -0.25%

# Pero el EA usa por defecto:
input double InpUmbralCompra = 0.0015;   # 0.0015% ← 167x más sensible
input double InpUmbralVenta  = -0.0015;  # -0.0015%
```

### 1.2 BUG: Contaminación temporal en el split Train/Test
- **Archivo**: `3_entrenar_modelo.py:255-264`
- **Problema**: Los datos de los 4 símbolos se apilan secuencialmente con `np.vstack()` y luego se hace un split 80/20. Esto significa que:
  - El 80% de entrenamiento contiene el 100% de XAUUSD + 100% de EURUSD + 100% de GBPUSD + parte de USDJPY
  - El 20% de test contiene solo el final de USDJPY
  - **No hay evaluación temporal real por símbolo**

```python
# Orden actual de apilado:
all_X = [XAUUSD_features, EURUSD_features, GBPUSD_features, USDJPY_features]
X_total = np.vstack(all_X)  # [~120K, ~120K, ~120K, ~120K] = ~480K filas

idx = int(len(X_total) * 0.80)  # ≈ 384,000
# Train: todo XAUUSD + todo EURUSD + todo GBPUSD + 80% USDJPY
# Test: solo 20% USDJPY ← ¡NO SE EVALÚAN LOS DEMÁS SÍMBOLOS!
```

- **Consecuencia**: La "precisión direccional" reportada solo refleja el rendimiento en USDJPY, no en los 4 activos

### 1.3 BUG: El modelo no distingue entre activos
- **Archivo**: `3_entrenar_modelo.py:218-248`
- **Problema**: No existe un feature de identificación de símbolo. El modelo recibe 20 features idénticos sin saber si está prediciendo XAUUSD (commodity) o EURUSD (forex)
- **Impacto**: El modelo aprende un promedio general que beneficia al activo dominante (XAUUSD, cuyo precio ~2000 genera returns con magnitudes diferentes a EURUSD ~1.10)

### 1.4 BUG: Escalas de precio incompatibles entre activos
- **Problema**: Los features basados en porcentajes (ret_1..ret_20, ema_dist, atr_pct) tienen distribuciones estadísticas muy diferentes:
  - XAUUSD: ATR diario ~$30 → atr_pct ≈ 0.015 (1.5%)
  - EURUSD: ATR diario ~0.006 → atr_pct ≈ 0.005 (0.5%)
  - USDJPY: ATR diario ~1.5 → atr_pct ≈ 0.010 (1.0%)
- El `StandardScaler` normaliza globalmente, mezclando distribuciones incompatibles

---

## 2. DEFICIENCIAS DE DISEÑO (Impacto en rendimiento subóptimo)

### 2.1 Sin backtesting Python independiente
- **Problema**: No existe un motor de backtesting en Python. Todo depende del Strategy Tester de MT5
- **Impacto**: No se puede iterar rápidamente, no se calculan métricas financieras, no se puede automatizar la optimización
- **Métricas ausentes**: Sharpe Ratio, Max Drawdown, Calmar Ratio, Profit Factor, Win Rate, Average Win/Loss

### 2.2 Sin validación Walk-Forward
- **Archivo**: `3_entrenar_modelo.py:261-264`
- **Problema**: Split estático 80/20 con un solo punto de corte temporal
- **Impacto**: Alta probabilidad de overfitting. No simula el despliegue real donde el modelo se entrena con datos pasados y opera en datos futuros no vistos

### 2.3 Sin evaluación por símbolo individual
- **Problema**: Solo se reportan métricas agregadas. No se sabe la precisión direccional de cada activo
- **Impacto**: Imposible diagnosticar qué activos funcionan y cuáles no

### 2.4 Modelo de regresión usado como clasificador
- **Archivo**: `3_entrenar_modelo.py:300-302`
- **Problema**: El modelo predice retornos continuos (regresión), pero se evalúa y usa solo por su signo (clasificación binaria)
- **Impacto**: Se desperdicia capacidad del modelo. Un clasificador binario o un modelo con umbral óptimo sería más eficiente

### 2.5 Sin gestión de correlación entre posiciones
- **Archivo**: `HybridAI_EA.mq5:186-202`
- **Problema**: El EA puede abrir posiciones simultáneas en los 4 símbolos correlacionados (EURUSD y GBPUSD correlación ~0.85)
- **Impacto**: Riesgo real duplicado cuando activos correlacionados se mueven juntos

### 2.6 Sin trailing stop ni gestión dinámica de posición
- **Archivo**: `HybridAI_EA.mq5:333-372`
- **Problema**: SL/TP fijos al momento de entrada. No hay trailing stop, breakeven, ni cierre parcial
- **Impacto**: Se dejan ganancias sobre la mesa en tendencias fuertes

### 2.7 El feature `vol_ratio` usa tick_volume, no volumen real
- **Archivo**: `2_descargar_datos.py:84-86`
- **Problema**: MT5 solo provee tick_volume para forex. El volumen real no está disponible
- **Impacto**: Feature 15 (vol_ratio) tiene significado diferente para cada activo y broker

---

## 3. DEFICIENCIAS DE DATOS Y PREPROCESAMIENTO

### 3.1 Sin detección de gaps temporales
- **Archivo**: `2_descargar_datos.py:81-92`
- **Problema**: No se verifican ni manejan gaps en los datos (fines de semana, festivos, cortes del broker)
- **Impacto**: Los indicadores técnicos calculan valores incorrectos a través de gaps (ej: retorno de 20 barras que cruza un fin de semana)

### 3.2 Sin filtrado de outliers estadísticos
- **Problema**: Solo se clipean los targets a [-10%, +10%] pero no se filtran features anómalos
- **Impacto**: Flash crashes, gaps de apertura y errores de datos contaminan el entrenamiento

### 3.3 Sin verificación de duplicados de timestamp
- **Archivo**: `2_descargar_datos.py:81-92`
- **Problema**: No se verifican timestamps duplicados que podrían existir en los datos de MT5

### 3.4 Descarga limitada a 120,000 barras
- **Archivo**: `2_descargar_datos.py:73-74`
- **Problema**: Algunos brokers limitan a menos barras. No hay paginación ni verificación del rango real obtenido

---

## 4. DEFICIENCIAS EN EL EA (MQL5)

### 4.1 ORDER_FILLING_IOC puede fallar según broker
- **Archivo**: `HybridAI_EA.mq5:69`
- **Problema**: `ORDER_FILLING_IOC` no es soportado por todos los brokers. Debería detectarse automáticamente
- **Impacto**: El EA puede fallar silenciosamente al abrir órdenes en algunos brokers

### 4.2 Sin protección contra slippage extremo
- **Problema**: El desvío de 30 puntos (línea 68) puede ser insuficiente en XAUUSD durante alta volatilidad
- **Impacto**: Ejecuciones a precios muy diferentes del esperado

### 4.3 Sin cierre de posiciones contrarias
- **Problema**: Si hay una señal de compra con una venta abierta, no se cierra la venta primero
- **Impacto**: Posiciones opuestas pueden coexistir drenando capital en spreads

### 4.4 Sin verificación de spread antes de operar
- **Problema**: No se verifica si el spread actual es aceptable antes de abrir una operación
- **Impacto**: Operaciones durante spread alto (noticias, bajo volumen) reducen el edge del modelo

---

# PARTE 2: PLAN DE OPTIMIZACIÓN PARA BACKTESTING MULTI-ACTIVO

## Fase 1: Corrección de Errores Críticos

### 1.1 Corregir split temporal por símbolo
```
ANTES: vstack(todos) → split 80/20 global
DESPUÉS: Para cada símbolo → split 80/20 individual → combinar trains / combinar tests
```

**Implementación**:
- Hacer el split 80/20 temporal DENTRO de cada símbolo antes de combinar
- Esto garantiza que cada activo tenga datos de test en su período más reciente
- Evaluar métricas por símbolo individual + métricas agregadas

### 1.2 Agregar identificador de símbolo como feature
- Añadir 4 features binarios (one-hot encoding): `is_xauusd, is_eurusd, is_gbpusd, is_usdjpy`
- Alternativa: 1 feature ordinal con ID numérico (0, 1, 2, 3)
- El modelo podrá aprender patrones específicos de cada activo
- **N_FEATURES pasa de 20 a 24** (o 21 con ordinal)

### 1.3 Sincronizar umbrales Config ↔ EA
- Usar los umbrales del config JSON como fuente de verdad
- Actualizar los defaults del EA para que coincidan
- Mejor aún: calcular umbrales óptimos por símbolo durante el entrenamiento

---

## Fase 2: Motor de Backtesting en Python

### 2.1 Crear `4_backtesting.py` — Motor de simulación de trading
**Componentes**:

```
BacktestEngine:
├── Simulación barra-a-barra (M15)
├── Gestión de posiciones (open/close/SL/TP)
├── Cálculo de slippage y spread
├── Cálculo de comisiones
└── Registro de operaciones (trade log)

Métricas a calcular:
├── Retorno total y anualizado
├── Sharpe Ratio (anualizado)
├── Sortino Ratio
├── Max Drawdown (% y duración)
├── Calmar Ratio (retorno / max DD)
├── Profit Factor (gross profit / gross loss)
├── Win Rate (% operaciones ganadoras)
├── Average Win / Average Loss
├── Expectancia por trade
├── Número total de trades
├── Trades por mes
└── Curva de equity
```

### 2.2 Backtesting por símbolo individual
- Ejecutar el backtest independientemente para cada uno de los 4 activos
- Comparar rendimiento: XAUUSD vs EURUSD vs GBPUSD vs USDJPY
- Identificar qué activos son rentables y cuáles no
- Generar gráficos de equity por activo

### 2.3 Backtesting con costos realistas
- Incluir spreads típicos por activo:
  - EURUSD: ~1.0 pips
  - GBPUSD: ~1.5 pips
  - USDJPY: ~1.2 pips
  - XAUUSD: ~30 cents (~3 pips)
- Incluir comisiones según broker
- Incluir slippage estimado (1-2 pips adicionales)

---

## Fase 3: Validación Walk-Forward

### 3.1 Implementar Walk-Forward Analysis
```
Ventana de entrenamiento: 3 años (deslizante)
Ventana de validación: 3 meses
Ventana de test: 3 meses
Paso de avance: 3 meses

Iteración 1: Train[2019-2022] → Val[2022-Q1] → Test[2022-Q2]
Iteración 2: Train[2019.Q2-2022.Q2] → Val[2022-Q3] → Test[2022-Q4]
Iteración 3: Train[2020-2023] → Val[2023-Q1] → Test[2023-Q2]
...etc
```

### 3.2 Validación cruzada temporal (TimeSeriesSplit)
- Usar `sklearn.model_selection.TimeSeriesSplit` con 5 folds
- Evaluar estabilidad del modelo a través del tiempo
- Detectar degradación del modelo en períodos específicos

### 3.3 Análisis de robustez del modelo
- Evaluar degradación de precisión mes a mes
- Identificar regímenes de mercado donde el modelo falla
- Medir la vida útil del modelo antes de reentrenamiento

---

## Fase 4: Mejoras al Modelo de IA

### 4.1 Normalización por símbolo (en lugar de global)
```python
# ANTES (problemático):
X_total = np.vstack(all_X)  # Mezcla escalas
pipeline = Pipeline([("scaler", StandardScaler()), ...])

# DESPUÉS (correcto):
# Opción A: Normalizar cada símbolo individualmente antes de combinar
for sym in symbols:
    X_sym = calcular_features(df_sym)
    X_sym_normalized = StandardScaler().fit_transform(X_sym)  # Escala propia
    all_X.append(X_sym_normalized)
# No usar StandardScaler en el pipeline final

# Opción B: Agregar símbolo como feature y que el modelo aprenda las escalas
```

### 4.2 Optimización de hiperparámetros
```python
param_grid = {
    'model__n_estimators': [300, 500, 800],
    'model__max_depth': [8, 12, 16, 20],
    'model__min_samples_leaf': [5, 10, 20, 50],
    'model__max_features': ['sqrt', 'log2', 0.5],
}
# Usar TimeSeriesSplit para cross-validation temporal
```

### 4.3 Probar modelos alternativos
- **LightGBM**: Más rápido, mejor manejo de features heterogéneos, soporta ONNX
- **XGBoost**: Más estable, mejor regularización
- **Ensemble**: Combinar múltiples modelos con votación ponderada

### 4.4 Nuevos features específicos por activo
```
Features adicionales propuestos:
├── Hora del día (cíclico: sin/cos) — sesiones de trading
├── Día de la semana (cíclico: sin/cos)
├── Volatilidad realizada (20 barras)
├── Skewness de retornos (20 barras)
├── Momentum de volumen (cambio % volumen 5 barras)
├── Spread medio (si disponible)
├── Correlación rolling con DXY (índice dólar) — feature externo
└── Regime indicator (alta/baja volatilidad basado en percentil ATR)
```

### 4.5 Selección de features por importancia
```python
# Evaluar importancia de features con el modelo entrenado
importances = pipeline.named_steps['model'].feature_importances_
# Eliminar features con importancia < threshold
# Reducir dimensionalidad mejora generalización
```

---

## Fase 5: Optimización de Parámetros de Trading

### 5.1 Optimización de umbrales por símbolo
```python
# Para cada símbolo, buscar el umbral óptimo:
for threshold in np.arange(0.05, 0.50, 0.01):
    trades = simulate(predictions, threshold)
    sharpe = calculate_sharpe(trades)
    # Encontrar threshold que maximiza Sharpe Ratio
```

### 5.2 Optimización de SL/TP por símbolo
```
Parámetros a optimizar (por activo):
├── SL_ATR_Mult: [1.0, 1.5, 2.0, 2.5, 3.0]
├── TP_ATR_Mult: [1.5, 2.0, 3.0, 4.0, 5.0]
├── Max trades simultáneos: [1, 2, 3]
└── Filtro de hora: por sesión de mercado
```

### 5.3 Perfiles de trading por activo
```
XAUUSD (Commodity):
├── Mayor volatilidad → SL más amplio (2.5× ATR)
├── Sesión óptima: Londres + NY overlap (13:00-17:00 GMT)
├── Umbral más alto (requiere señal más fuerte)
└── TP/SL ratio: 2:1 a 3:1

EURUSD (Major FX):
├── Volatilidad media → SL estándar (2.0× ATR)
├── Sesión óptima: Londres (07:00-16:00 GMT)
├── Umbral medio
└── TP/SL ratio: 1.5:1 a 2.5:1

GBPUSD (Major FX):
├── Alta volatilidad → SL adaptativo
├── Sesión óptima: Londres (07:00-16:00 GMT)
├── Sensible a noticias UK → filtrar eventos
└── TP/SL ratio: 2:1 a 3:1

USDJPY (Yen cross):
├── Menor volatilidad → SL más estrecho (1.5× ATR)
├── Sesión óptima: Tokio + Londres (00:00-10:00 GMT)
├── Comportamiento trending → beneficia trailing stop
└── TP/SL ratio: 2:1
```

---

## Fase 6: Mejoras al EA (MQL5)

### 6.1 Detección automática de filling mode
```cpp
// Detectar el modo de llenado soportado por el broker
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
    long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
    if(filling & SYMBOL_FILLING_FOK) return ORDER_FILLING_FOK;
    if(filling & SYMBOL_FILLING_IOC) return ORDER_FILLING_IOC;
    return ORDER_FILLING_RETURN;
}
```

### 6.2 Filtro de spread
```cpp
// No operar si el spread supera 2× el spread medio
double spread_actual = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
double spread_max_permitido = atr_val * 0.1;  // 10% del ATR como máximo
if(spread_actual > spread_max_permitido) return;
```

### 6.3 Trailing Stop basado en ATR
```cpp
// Cada barra, ajustar SL si la posición está en ganancia
void GestionarTrailingStop()
{
    // Mover SL a breakeven + X cuando ganancia > 1× ATR
    // Trail stop a distancia de 1.5× ATR del precio actual
}
```

### 6.4 Cierre de posición contraria antes de nueva entrada
```cpp
// Si señal BUY y hay SELL abierta → cerrar SELL primero
if(prediccion > InpUmbralCompra && n_sell > 0)
    CerrarPosiciones(POSITION_TYPE_SELL);
```

---

## Fase 7: Pipeline de Ejecución

### Orden de implementación recomendado:

```
SEMANA 1: Correcciones críticas
├── [1] Corregir split temporal por símbolo
├── [2] Sincronizar umbrales Config/EA
├── [3] Agregar evaluación por símbolo individual
└── [4] Crear reporte de métricas por activo

SEMANA 2: Motor de backtesting
├── [5] Implementar 4_backtesting.py con simulación de trading
├── [6] Agregar costos realistas (spread + comisiones)
├── [7] Calcular métricas financieras completas
└── [8] Generar gráficos de equity y drawdown

SEMANA 3: Mejoras al modelo
├── [9] Agregar symbol_id como feature
├── [10] Implementar normalización por símbolo
├── [11] Walk-forward validation
├── [12] Optimización de hiperparámetros

SEMANA 4: Optimización de trading
├── [13] Optimizar umbrales por símbolo
├── [14] Optimizar SL/TP por símbolo
├── [15] Agregar nuevos features (hora, día, volatilidad)
├── [16] Mejorar EA (trailing stop, spread filter, filling mode)

SEMANA 5: Validación final
├── [17] Ejecutar backtesting completo en los 4 activos
├── [18] Walk-forward con datos out-of-sample
├── [19] Comparar rendimiento modelo optimizado vs original
└── [20] Documentar resultados y configuración final
```

---

## Criterios de Éxito para el Nuevo Backtesting

| Métrica | Mínimo Aceptable | Objetivo |
|---------|-------------------|----------|
| Precisión direccional (cada activo) | >53% | >56% |
| Sharpe Ratio (anual) | >0.8 | >1.5 |
| Max Drawdown | <20% | <12% |
| Profit Factor | >1.2 | >1.5 |
| Win Rate | >48% | >55% |
| Trades/mes (por activo) | >10 | 20-40 |
| Calmar Ratio | >0.5 | >1.0 |
| Consistencia walk-forward | >60% períodos rentables | >75% |

---

## Resumen de Archivos a Crear/Modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `3_entrenar_modelo.py` | MODIFICAR | Corregir split, agregar symbol_id, eval por símbolo |
| `4_backtesting.py` | CREAR | Motor de backtesting con métricas financieras |
| `5_optimizar_parametros.py` | CREAR | Optimización de hiperparámetros y trading params |
| `6_walk_forward.py` | CREAR | Validación walk-forward |
| `HybridAI_EA.mq5` | MODIFICAR | Trailing stop, spread filter, filling mode |
| `config/symbols.json` | CREAR | Configuración específica por activo |
| `requirements.txt` | MODIFICAR | Agregar matplotlib, lightgbm, optuna |
