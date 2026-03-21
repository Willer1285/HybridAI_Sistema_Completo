# Guia Paso a Paso: Implementacion y Prueba de HybridAI v2.0

## Requisitos Previos

### Software necesario
- **Windows 10/11** (MetaTrader 5 solo corre en Windows)
- **Python 3.8 - 3.11** (3.12+ puede tener problemas con MetaTrader5)
- **MetaTrader 5** instalado y con una cuenta (demo o real)
- **MetaEditor** (viene incluido con MT5)

### Cuenta de broker
- Abre una **cuenta DEMO** con un broker que soporte los 4 activos:
  - XAUUSD (Gold)
  - EURUSD
  - GBPUSD
  - USDJPY
- Brokers recomendados para demo: IC Markets, Pepperstone, FP Markets, RoboForex
- Asegurate de que la cuenta tenga **apalancamiento 1:100** o superior
- Balance recomendado para demo: **$10,000 USD**

---

## PARTE A: ENTRENAMIENTO DEL MODELO (Python)

### Paso 1: Preparar el entorno Python

```bash
# Crear entorno virtual (recomendado)
python -m venv hybridai_env
hybridai_env\Scripts\activate

# Instalar dependencias
cd HybridAI_Sistema_Completo
pip install -r requirements.txt
```

**Verificar instalacion:**
```bash
python 1_verificar_instalacion.py
```

Deberias ver:
```
  MetaTrader5     -> OK (version X.X.X)
  pandas          -> OK
  numpy           -> OK
  scikit-learn    -> OK
  skl2onnx        -> OK
  onnx            -> OK
  onnxruntime     -> OK
  Conexion MT5    -> OK (Broker: Tu Broker)
```

**Si falla la conexion MT5:**
1. Abre MetaTrader 5 y espera a que se conecte al servidor
2. Asegurate de estar logueado en una cuenta (demo o real)
3. Ve a Herramientas > Opciones > Expert Advisors > Marca "Permitir Algo Trading"
4. Vuelve a ejecutar el script

**Si falla alguna libreria:**
```bash
pip install --upgrade MetaTrader5 pandas numpy scikit-learn skl2onnx onnx onnxruntime
```

---

### Paso 2: Descargar datos historicos

**IMPORTANTE: MetaTrader 5 DEBE estar abierto y conectado.**

```bash
python 2_descargar_datos.py
```

Deberias ver algo como:
```
  Descargando XAUUSD...
  115,342 barras guardadas -> datos/xauusd_m15.csv
  Limpieza: 12 filas removidas (3 dups, 2 outliers)

  Descargando EURUSD...
  118,205 barras guardadas -> datos/eurusd_m15.csv
  ...
```

**Si un simbolo no se encuentra:**
1. Abre MT5, ve a "Ver > Simbolos" (Ctrl+U)
2. Busca el nombre exacto del simbolo (puede ser "GOLD" en vez de "XAUUSD")
3. Edita `2_descargar_datos.py` linea 16-21 y agrega la variante de tu broker

**Verificar los datos:**
- Revisa que la carpeta `datos/` contenga 4 archivos CSV
- Cada archivo debe tener >80,000 barras (idealmente >100,000)
- Si alguno tiene menos de 50,000, tu broker limita el historial
  - Solucion: Descarga datos desde otra fuente o usa un broker con mas historial

---

### Paso 3: Entrenar el modelo

```bash
python 3_entrenar_modelo.py
```

**Duracion estimada:** 3-10 minutos segun tu PC

Deberias ver:
```
  RESULTADOS GLOBALES
  Precision direccional (%)     Train    Test
                                 XX.X    XX.X

  RESULTADOS POR SIMBOLO
  Simbolo      RMSE     MAE    Dir%   Muestras
  XAUUSD     0.XXXX  0.XXXX   XX.X     XX,XXX
  EURUSD     0.XXXX  0.XXXX   XX.X     XX,XXX
  GBPUSD     0.XXXX  0.XXXX   XX.X     XX,XXX
  USDJPY     0.XXXX  0.XXXX   XX.X     XX,XXX

  UMBRALES OPTIMOS POR SIMBOLO
  XAUUSD: umbral = 0.XX%
  ...

  IMPORTANCIA DE FEATURES (Top 10)
  ...
```

**Que buscar:**
- Precision direccional global test > 52% (mejor que azar)
- Cada simbolo individualmente > 51%
- RMSE test < 0.10
- Si algun simbolo tiene precision < 50%, puede ser no rentable

**Archivos generados:**
- `modelo/hybrid_ai_model.onnx` - El modelo para MT5
- `modelo/modelo_config.json` - Configuracion y umbrales

---

### Paso 4: Ejecutar backtesting

```bash
python 4_backtesting.py
```

**Duracion estimada:** 2-5 minutos

Deberias ver:
```
  BACKTESTING: XAUUSD
  Periodo test: 2024-XX-XX -> 2026-XX-XX
  Total trades:      XXX
  Profit neto:       $+XXX.XX
  Win Rate:          XX.X%
  Sharpe Ratio:      X.XX
  Max Drawdown:      X.XX%
  Profit Factor:     X.XX
  ...

  RESUMEN COMPARATIVO - TODOS LOS SIMBOLOS
  Simbolo  Trades   Ret%  WinR%    PF  Sharpe  MaxDD%  $/trade
  XAUUSD     XXX  +XX.X   XX.X  X.XX    X.XX    X.XX    +X.XX
  EURUSD     XXX  +XX.X   XX.X  X.XX    X.XX    X.XX    +X.XX
  ...
```

**Criterios de aceptacion:**
| Metrica | Minimo | Bueno | Excelente |
|---------|--------|-------|-----------|
| Sharpe Ratio | > 0.5 | > 1.0 | > 1.5 |
| Max Drawdown | < 25% | < 15% | < 10% |
| Profit Factor | > 1.1 | > 1.3 | > 1.5 |
| Win Rate | > 45% | > 50% | > 55% |
| Expectancia | > $0 | > $5 | > $15 |

**Si los resultados son malos:**
- Revisa que los datos tengan suficiente historial
- Ejecuta paso 6 para optimizar parametros
- Considera desactivar simbolos con Sharpe negativo

---

### Paso 5: Validacion Walk-Forward

```bash
python 5_walk_forward.py
```

**Duracion estimada:** 15-45 minutos (entrena multiples modelos)

Deberias ver:
```
  WALK-FORWARD: XAUUSD
  Fold  1: Dir=XX.X%  RMSE=0.XXXX  Trades=XXX  Ret=+X.XXX%  [OK]
  Fold  2: Dir=XX.X%  RMSE=0.XXXX  Trades=XXX  Ret=+X.XXX%  [OK]
  ...
  Resumen XAUUSD
  Folds rentables: X/Y (XX%)
```

**Que buscar:**
- Consistencia > 60% (al menos 60% de los periodos son rentables)
- Si consistencia < 50%, el modelo no es estable temporalmente
- Precision media > 52%

---

### Paso 6: Optimizacion de parametros (opcional pero recomendado)

```bash
python 6_optimizar_parametros.py
```

**Duracion estimada:** 30-90 minutos

Esto busca la mejor combinacion de:
- Hiperparametros del modelo (profundidad, arboles, hojas minimas)
- Parametros de trading (umbrales, SL/TP por simbolo)

Los resultados se guardan en:
- `modelo/optimal_params.json`
- `config/symbols.json` (actualizado con parametros optimos)

---

## PARTE B: DESPLIEGUE EN METATRADER 5

### Paso 7: Copiar el modelo ONNX a MT5

1. Localiza el archivo: `modelo/hybrid_ai_model.onnx`
2. En MT5, ve a **Archivo > Abrir Carpeta de Datos** (File > Open Data Folder)
3. Navega a `MQL5\Files\`
4. Copia `hybrid_ai_model.onnx` ahi

**Ruta tipica:**
```
C:\Users\TuUsuario\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Files\
```

### Paso 8: Compilar el EA

1. Abre **MetaEditor** (F4 desde MT5, o desde el menu Herramientas)
2. Archivo > Abrir > Navega hasta `HybridAI_EA.mq5`
   - Si prefieres, copia `HybridAI_EA.mq5` a `MQL5\Experts\` y abrelo desde ahi
3. Presiona **Compilar** (F7) o el boton "Compilar"
4. Verifica: **0 errores, 0 warnings** en la ventana de salida

**Si hay errores de compilacion:**
- Error "Trade.mqh not found": Verifica que tengas MT5 actualizado
- Error "OnnxCreate": Tu version de MT5 necesita Build 3500+ para soporte ONNX

---

## PARTE C: BACKTESTING EN MT5 (Strategy Tester)

### Paso 9: Configurar el Strategy Tester

1. En MT5, ve a **Ver > Probador de Estrategias** (Ctrl+R)
2. Configura:

| Parametro | Valor |
|-----------|-------|
| **Expert Advisor** | HybridAI_EA |
| **Simbolo** | XAUUSD (empezar con este) |
| **Periodo** | M15 |
| **Fecha Inicio** | 2024.01.01 |
| **Fecha Fin** | 2026.01.01 (o fecha actual) |
| **Modelo** | Cada tick basado en ticks reales (mas preciso) |
| **Deposito** | 10000 |
| **Apalancamiento** | 1:100 |
| **Optimizacion** | Deshabilitada (primera prueba) |

3. Click en **Configuracion** (la rueda dentada) para ajustar parametros del EA:

**Parametros recomendados para primera prueba:**
| Parametro | XAUUSD | EURUSD | GBPUSD | USDJPY |
|-----------|--------|--------|--------|--------|
| InpUmbralCompra | 0.20 | 0.15 | 0.15 | 0.10 |
| InpUmbralVenta | -0.20 | -0.15 | -0.15 | -0.10 |
| InpSL_ATR_Mult | 2.5 | 2.0 | 2.0 | 1.5 |
| InpTP_ATR_Mult | 4.0 | 3.0 | 3.5 | 2.5 |
| InpRiesgoPct | 1.0 | 1.0 | 1.0 | 1.0 |
| InpTrailingStop | true | true | true | true |
| InpFiltroSpread | true | true | true | true |

4. Click en **Iniciar** para ejecutar el backtest

### Paso 10: Analizar resultados del backtest MT5

Cuando termine, revisa las pestanas:

**Pestana "Resultados":**
- Profit neto total
- Total de trades
- Profit Factor
- Recovery Factor
- Sharpe Ratio

**Pestana "Grafico":**
- La curva de equity debe ser ascendente y suave
- Evitar caidas bruscas (drawdowns prolongados)

**Pestana "Backtest":**
- Max Drawdown (debe ser < 20% del balance)
- Win Rate (idealmente > 48%)
- Average Win vs Average Loss (avg win debe ser > avg loss)

**Repetir para cada simbolo:**
- Cambia el simbolo en el tester y ejecuta de nuevo
- Compara resultados entre los 4 activos
- Identifica cuales son rentables y cuales no

### Paso 11: Optimizacion en MT5 (opcional)

1. En el Strategy Tester, cambia **Optimizacion** a "Completa"
2. En Configuracion, marca los parametros a optimizar:
   - InpUmbralCompra: Start=0.05, Step=0.05, Stop=0.40
   - InpSL_ATR_Mult: Start=1.5, Step=0.5, Stop=3.5
   - InpTP_ATR_Mult: Start=2.0, Step=0.5, Stop=5.0
3. Click en Iniciar
4. Revisa la pestana "Resultados de Optimizacion"
5. Ordena por Sharpe Ratio o Profit Factor
6. Selecciona la mejor combinacion

---

## PARTE D: PRUEBA EN CUENTA DEMO (Paper Trading)

### Paso 12: Preparar la cuenta demo

1. **Abre una cuenta demo** si aun no tienes una
2. Configuracion recomendada:
   - Balance: $10,000
   - Apalancamiento: 1:100
   - Sin comisiones si es posible (o comisiones estandar)
3. Asegurate de que los 4 simbolos estan visibles en Market Watch

### Paso 13: Cargar el EA en un grafico

**Para cada simbolo que quieras operar:**

1. Abre un grafico del simbolo (ej: XAUUSD)
2. Cambia el timeframe a **M15**
3. Arrastra el EA `HybridAI_EA` desde el Navigator al grafico
4. En la ventana de configuracion:
   - Pestana **Comun**:
     - Marca "Permitir trading en vivo"
     - Marca "Permitir importar DLL" (no necesario pero util)
   - Pestana **Inputs**:
     - Ajusta los parametros segun la tabla del Paso 9
     - Asegurate de que `InpModelFile = hybrid_ai_model.onnx`
5. Click OK

**Verificar que el EA esta activo:**
- Deberia aparecer "HybridAI_EA" en la esquina superior derecha del grafico
- En la pestana "Expertos" (abajo), deberia ver el log de inicio:
  ```
  HybridAI EA v2.0 - Sistema de Trading con IA
  Simbolo: XAUUSD (ID=0)
  Modelo: hybrid_ai_model.onnx -> CARGADO (24 features)
  ...
  ```

### Paso 14: Verificar que opera correctamente

1. **Espera** a que se generen senales (puede tomar horas o dias)
2. Revisa la pestana "Expertos" periodicamente para ver:
   ```
   XAUUSD | 2026.03.21 14:00 | Prediccion IA: 0.2341%
   >> COMPRA | Pred=0.2341% | Precio=2985.50 | SL=2980.20 | TP=2996.10 | Lotes=0.05
   ```
3. Revisa la pestana "Trade" para ver posiciones abiertas
4. Revisa la pestana "Historial" para ver trades cerrados

### Paso 15: Monitoreo durante la demo

**Primer dia:**
- Verifica que el EA genera predicciones cada 10 barras (log)
- Verifica que abre trades cuando la prediccion supera el umbral
- Verifica que el SL/TP se colocan correctamente
- Verifica que el trailing stop mueve el SL (si esta activado)

**Primera semana:**
- Revisa cuantos trades ha abierto por simbolo
- Si un simbolo no opera en 3 dias, el umbral puede ser muy alto -> bajalo
- Si opera demasiado (>10 trades/dia), el umbral es muy bajo -> subelo

**Primer mes (periodo minimo de evaluacion):**
- Genera un reporte: Click derecho en Historial > Reporte
- Compara con los resultados del backtesting Python
- Evalua:
  - Si los resultados son similares al backtest: el modelo es estable
  - Si son significativamente peores: posible overfitting
  - Si son mejores: suerte o condiciones favorables (no extrapolar)

---

## PARTE E: CRITERIOS DE DECISION

### Cuando pasar de demo a real

**NO pasar a real hasta cumplir TODOS estos criterios:**

1. Minimo **3 meses** de demo rentable
2. Max drawdown en demo < 15%
3. Profit Factor > 1.2 en los ultimos 2 meses
4. Al menos 50 trades ejecutados por simbolo
5. Resultados de demo consistentes con backtest (+/-30%)
6. Sin errores de ejecucion en el log del EA

### Cuando desactivar un simbolo

Desactiva un simbolo si en demo muestra:
- Sharpe Ratio < 0 despues de 1 mes
- Win Rate < 40% con >30 trades
- Drawdown > 20% en un solo simbolo
- Expectancia por trade negativa

### Cuando reentrenar el modelo

Reentrena (ejecutar pasos 2-6 de nuevo) cuando:
- Han pasado 3-6 meses desde el ultimo entrenamiento
- La precision direccional cae por debajo del 50% en algun simbolo
- Cambio de regimen de mercado evidente (nueva politica monetaria, crisis)
- El profit factor mensual baja consistentemente durante 2+ meses

---

## Resolucion de Problemas Comunes

| Problema | Solucion |
|----------|----------|
| "No se pudo cargar ONNX" | Verifica que el .onnx esta en MQL5/Files/ |
| "Error configurando input shape" | El modelo tiene 24 features, no 20. Reentrena con v2.0 |
| EA no opera | Verifica: Algo Trading ON, timeframe M15, umbrales no muy altos |
| Trades con SL muy ajustado | Aumenta InpSL_ATR_Mult (prueba 2.5 o 3.0) |
| Demasiados trades | Aumenta umbrales (InpUmbralCompra/Venta) |
| Pocos trades | Baja umbrales o desactiva filtro de hora |
| "OnnxRun fallo" | Modelo corrupto, reentrena y reemplaza el .onnx |
| Spread filter bloquea todo | Baja InpMaxSpreadATR a 0.15 o desactiva filtro |
| Filling mode error | El EA autodetecta, pero si falla, cambia broker |
| Error al compilar | Actualiza MT5 a Build 3500+ (menu Ayuda > Buscar actualizacion) |

---

## Diagrama de Flujo Completo

```
PYTHON (Entrenamiento)              METATRADER 5 (Ejecucion)
========================            ==========================

1_verificar_instalacion.py
         |
2_descargar_datos.py
    (genera CSVs)
         |
3_entrenar_modelo.py
    (genera .onnx)
         |
4_backtesting.py ---------> Comparar con Strategy Tester
    (metricas)
         |
5_walk_forward.py
    (estabilidad)
         |
6_optimizar_parametros.py
    (parametros optimos)
         |                          Copiar .onnx a MQL5/Files/
         |                                   |
         |                          Compilar HybridAI_EA.mq5
         |                                   |
         |                          Strategy Tester (backtest)
         |                                   |
         +-- Comparar resultados ---+        |
                                    |   Cuenta Demo (1-3 meses)
                                    |        |
                                    +-- Evaluar consistencia
                                             |
                                        Cuenta Real
                                   (solo si criterios cumplidos)
```
