//+------------------------------------------------------------------+
//|                                             HybridAI_EA.mq5     |
//|          Sistema de Trading con Inteligencia Artificial          |
//|          Modelo: ExtraTrees + ONNX | Timeframe: M15              |
//|          Instrumentos: XAU/USD, EUR/USD, GBP/USD, USD/JPY        |
//+------------------------------------------------------------------+
#property copyright   "HybridAI Trading System 2026"
#property description "EA con modelo de IA (ONNX) para señales de trading"
#property version     "1.00"
#property strict
#property tester_file "hybrid_ai_model.onnx"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//====================================================================
//  PARÁMETROS DE ENTRADA
//====================================================================

input group "════ MODELO IA ════════════════════════════"
input string   InpModelFile      = "hybrid_ai_model.onnx";
//  ^── Nombre del archivo ONNX en la carpeta MQL5\Files\

input group "════ UMBRALES DE SEÑAL ════════════════════"
input double   InpUmbralCompra   =  0.0015;  // Comprar si predicción > este valor (%)
input double   InpUmbralVenta    = -0.0015;  // Vender  si predicción < este valor (%)

input group "════ GESTIÓN DE RIESGO ════════════════════"
input double   InpRiesgoPct      =  1.0;    // Riesgo por operación (% del balance)
input double   InpSL_ATR_Mult    =  2.0;    // Stop Loss  = ATR × este multiplicador
input double   InpTP_ATR_Mult    =  3.0;    // Take Profit= ATR × este multiplicador
input int      InpMaxTrades      =  1;      // Máximo de trades abiertos a la vez

input group "════ FILTROS OPCIONALES ═══════════════════"
input bool     InpFiltroHora     = false;   // Activar filtro de hora
input int      InpHoraInicio     =  7;      // Hora inicio (GMT)
input int      InpHoraFin        = 20;      // Hora fin   (GMT)

input group "════ IDENTIFICACIÓN ═══════════════════════"
input int      InpMagicNumber    = 246810;  // Número mágico del EA

//====================================================================
//  VARIABLES GLOBALES
//====================================================================

long      g_onnx     = INVALID_HANDLE;
CTrade    g_trade;

int       g_h_atr    = INVALID_HANDLE;
int       g_h_rsi    = INVALID_HANDLE;
int       g_h_macd   = INVALID_HANDLE;
int       g_h_bb     = INVALID_HANDLE;
int       g_h_ema9   = INVALID_HANDLE;
int       g_h_ema21  = INVALID_HANDLE;
int       g_h_ema50  = INVALID_HANDLE;

#define N_FEAT    20      // Total de features (debe coincidir con Python)
#define LOOKBACK  60      // Barras mínimas para calentar indicadores

//====================================================================
//  INICIALIZACIÓN
//====================================================================

int OnInit()
{
   //── Configurar objeto de trading ────────────────────────────
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   //── Cargar el modelo ONNX ───────────────────────────────────
   g_onnx = OnnxCreate(InpModelFile, ONNX_DEFAULT);

   if(g_onnx == INVALID_HANDLE)
   {
      Alert("❌ HybridAI: No se pudo cargar ONNX: ", InpModelFile,
            "\n\nVerifica que copiaste el archivo a:\n",
            "MT5 → File → Open Data Folder → MQL5 → Files");
      return INIT_FAILED;
   }

   //── Configurar shapes de entrada/salida del modelo ONNX ─────
   long sh_in[]  = {1, N_FEAT};   // [batch=1, features=20]
   long sh_out[] = {1, 1};        // [batch=1, outputs=1]  → un valor float de predicción

   if(!OnnxSetInputShape(g_onnx, 0, sh_in))
   {
      Alert("❌ HybridAI: Error configurando input shape. Código: ", GetLastError());
      return INIT_FAILED;
   }
   if(!OnnxSetOutputShape(g_onnx, 0, sh_out))
   {
      Alert("❌ HybridAI: Error configurando output shape. Código: ", GetLastError());
      return INIT_FAILED;
   }

   //── Crear handles de indicadores técnicos ───────────────────
   g_h_atr  = iATR (_Symbol, PERIOD_CURRENT, 14);
   g_h_rsi  = iRSI (_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   g_h_macd = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   g_h_bb   = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
   g_h_ema9  = iMA  (_Symbol, PERIOD_CURRENT,  9, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema21 = iMA  (_Symbol, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema50 = iMA  (_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);

   //── Validar handles ─────────────────────────────────────────
   if(g_h_atr  == INVALID_HANDLE || g_h_rsi  == INVALID_HANDLE ||
      g_h_macd == INVALID_HANDLE || g_h_bb   == INVALID_HANDLE ||
      g_h_ema9  == INVALID_HANDLE || g_h_ema21 == INVALID_HANDLE ||
      g_h_ema50 == INVALID_HANDLE)
   {
      Alert("❌ HybridAI: Error al crear indicadores. Código: ", GetLastError());
      return INIT_FAILED;
   }

   //── Log de inicio ────────────────────────────────────────────
   Print("╔════════════════════════════════════════╗");
   Print("║    HybridAI EA - Sistema de Trading    ║");
   Print("╚════════════════════════════════════════╝");
   Print("  Símbolo : ", _Symbol, "  |  Timeframe : ", EnumToString(PERIOD_CURRENT));
   Print("  Modelo  : ", InpModelFile, "  → CARGADO ✅");
   Print("  Umbral  : Compra >", InpUmbralCompra, "%  |  Venta <", InpUmbralVenta, "%");
   Print("  Riesgo  : ", InpRiesgoPct, "% por trade  |  SL=", InpSL_ATR_Mult, "×ATR  |  TP=", InpTP_ATR_Mult, "×ATR");

   return INIT_SUCCEEDED;
}

//====================================================================
//  LIBERACIÓN DE RECURSOS
//====================================================================

void OnDeinit(const int reason)
{
   if(g_onnx != INVALID_HANDLE) OnnxRelease(g_onnx);

   int handles[] = {g_h_atr, g_h_rsi, g_h_macd, g_h_bb, g_h_ema9, g_h_ema21, g_h_ema50};
   for(int i = 0; i < ArraySize(handles); i++)
      if(handles[i] != INVALID_HANDLE) IndicatorRelease(handles[i]);

   Print("HybridAI detenido.");
}

//====================================================================
//  TICK PRINCIPAL
//====================================================================

void OnTick()
{
   //── Solo actuar en una nueva barra completada ────────────────
   static datetime ultima_barra = 0;
   datetime barra_actual = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(barra_actual == ultima_barra) return;
   ultima_barra = barra_actual;

   //── Esperar suficientes barras para indicadores ──────────────
   if(Bars(_Symbol, PERIOD_CURRENT) < LOOKBACK) return;

   //── Filtro de hora (opcional) ────────────────────────────────
   if(InpFiltroHora)
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < InpHoraInicio || dt.hour >= InpHoraFin) return;
   }

   //── Calcular los 20 features ─────────────────────────────────
   matrixf features(1, N_FEAT);
   if(!CalcularFeatures(features)) return;

   //── Ejecutar inferencia ONNX ─────────────────────────────────
   vectorf salida(1);
   if(!OnnxRun(g_onnx, ONNX_DEFAULT, features, salida))
   {
      Print("⚠️  OnnxRun falló en ", TimeToString(barra_actual), " | Error: ", GetLastError());
      return;
   }

   double prediccion = (double)salida[0];

   //── Log cada 10 barras ───────────────────────────────────────
   static int cnt = 0;
   if(++cnt % 10 == 0)
      Print("📊 ", _Symbol, " | ", TimeToString(barra_actual, TIME_DATE|TIME_MINUTES),
            " | Predicción IA: ", DoubleToString(prediccion, 4), "%");

   //── Contar posiciones abiertas ───────────────────────────────
   int n_buy  = ContarPosiciones(POSITION_TYPE_BUY);
   int n_sell = ContarPosiciones(POSITION_TYPE_SELL);
   int n_tot  = n_buy + n_sell;

   //── SEÑAL DE COMPRA ──────────────────────────────────────────
   if(prediccion > InpUmbralCompra)
   {
      if(n_buy == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_BUY, prediccion);
   }
   //── SEÑAL DE VENTA ───────────────────────────────────────────
   else if(prediccion < InpUmbralVenta)
   {
      if(n_sell == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_SELL, prediccion);
   }
}

//====================================================================
//  CALCULAR 20 FEATURES  ← DEBE SER IDÉNTICO AL PYTHON
//====================================================================

bool CalcularFeatures(matrixf &feat)
{
   const int LB = 55;   // barras históricas a pedir

   //── Obtener OHLCV ────────────────────────────────────────────
   double close[], high[], low[];
   double open_arr[];
   long   vol[];

   ArraySetAsSeries(close,    true);
   ArraySetAsSeries(high,     true);
   ArraySetAsSeries(open_arr, true);
   ArraySetAsSeries(low,      true);
   ArraySetAsSeries(vol,      true);

   if(CopyClose     (_Symbol, PERIOD_CURRENT, 1, LB+5, close)    < LB) return false;
   if(CopyHigh      (_Symbol, PERIOD_CURRENT, 1, LB+5, high)     < LB) return false;
   if(CopyLow       (_Symbol, PERIOD_CURRENT, 1, LB+5, low)      < LB) return false;
   if(CopyOpen      (_Symbol, PERIOD_CURRENT, 1, LB+5, open_arr) < LB) return false;
   if(CopyTickVolume(_Symbol, PERIOD_CURRENT, 1, LB+5, vol)      < LB) return false;

   //── Obtener buffers de indicadores ───────────────────────────
   double atr[], rsi[], macd_m[], macd_s[], macd_h[];
   double bb_u[], bb_mid[], bb_l[];
   double ema9a[], ema21a[], ema50a[];

   ArraySetAsSeries(atr,    true);  ArraySetAsSeries(rsi,    true);
   ArraySetAsSeries(macd_m, true);  ArraySetAsSeries(macd_s, true);
   ArraySetAsSeries(macd_h, true);
   ArraySetAsSeries(bb_u,   true);  ArraySetAsSeries(bb_mid, true);
   ArraySetAsSeries(bb_l,   true);
   ArraySetAsSeries(ema9a,  true);  ArraySetAsSeries(ema21a, true);
   ArraySetAsSeries(ema50a, true);

   if(CopyBuffer(g_h_atr,   0, 1, LB, atr)    < LB) return false;
   if(CopyBuffer(g_h_rsi,   0, 1, LB, rsi)    < LB) return false;
   if(CopyBuffer(g_h_macd,  0, 1, LB, macd_m) < LB) return false;
   if(CopyBuffer(g_h_macd,  1, 1, LB, macd_s) < LB) return false;
   // El histograma MACD se calcula manualmente porque iMACD solo tiene 2 buffers (0 y 1) en MT5.
   if(CopyBuffer(g_h_bb,    1, 1, LB, bb_u)   < LB) return false;
   if(CopyBuffer(g_h_bb,    0, 1, LB, bb_mid) < LB) return false;
   if(CopyBuffer(g_h_bb,    2, 1, LB, bb_l)   < LB) return false;
   if(CopyBuffer(g_h_ema9,  0, 1, LB, ema9a)  < LB) return false;
   if(CopyBuffer(g_h_ema21, 0, 1, LB, ema21a) < LB) return false;
   if(CopyBuffer(g_h_ema50, 0, 1, LB, ema50a) < LB) return false;

   //── Valores de la barra más reciente completada (índice 0) ───
   double c0   = close[0];
   double h0   = high[0];
   double l0   = low[0];
   double o0   = open_arr[0];
   double atr0 = atr[0];

   if(atr0 <= 0.0 || c0 <= 0.0) return false;

   double bb_ancho = bb_u[0] - bb_l[0];
   double hl_rango = h0 - l0;

   //── Volumen medio 20 barras ────────────────────────────────
   double vol_sum = 0;
   for(int i = 0; i < 20; i++) vol_sum += (double)vol[i];
   double vol_ma = vol_sum / 20.0;

   //── Williams %R(14) ────────────────────────────────────────
   double max_h = high[0], min_l = low[0];
   for(int i = 1; i < 14; i++)
   {
      if(high[i] > max_h) max_h = high[i];
      if(low[i]  < min_l) min_l = low[i];
   }
   double willr_rng = max_h - min_l;

   //── Asignar features (misma lógica que calcular_features en Python) ──

   // 0  RSI normalizado
   feat[0][0]  = (float)(rsi[0] / 100.0);

   // 1-3  MACD normalizado por ATR
   feat[0][1]  = (float)(macd_m[0] / atr0);
   feat[0][2]  = (float)(macd_s[0] / atr0);
   feat[0][3]  = (float)((macd_m[0] - macd_s[0]) / atr0);

   // 4  ATR como % del precio
   feat[0][4]  = (float)(atr0 / c0);

   // 5-6  Bollinger Bands
   feat[0][5]  = (float)(bb_ancho > 0 ? (c0 - bb_l[0]) / bb_ancho : 0.5);
   feat[0][6]  = (float)(bb_mid[0]> 0 ? bb_ancho / bb_mid[0]       : 0.0);

   // 7-9  Distancia % a EMAs
   feat[0][7]  = (float)(ema9a[0]  > 0 ? (c0 - ema9a[0])  / ema9a[0]  * 100.0 : 0.0);
   feat[0][8]  = (float)(ema21a[0] > 0 ? (c0 - ema21a[0]) / ema21a[0] * 100.0 : 0.0);
   feat[0][9]  = (float)(ema50a[0] > 0 ? (c0 - ema50a[0]) / ema50a[0] * 100.0 : 0.0);

   // 10-14  Retornos históricos (%)
   // Nota: close[0]=barra reciente, close[N]=N barras atrás
   feat[0][10] = (float)(close[1]  > 0 ? (c0 - close[1])  / close[1]  * 100.0 : 0.0);
   feat[0][11] = (float)(close[3]  > 0 ? (c0 - close[3])  / close[3]  * 100.0 : 0.0);
   feat[0][12] = (float)(close[5]  > 0 ? (c0 - close[5])  / close[5]  * 100.0 : 0.0);
   feat[0][13] = (float)(close[10] > 0 ? (c0 - close[10]) / close[10] * 100.0 : 0.0);
   feat[0][14] = (float)(close[20] > 0 ? (c0 - close[20]) / close[20] * 100.0 : 0.0);

   // 15  Ratio de volumen
   feat[0][15] = (float)(vol_ma > 0 ? (double)vol[0] / vol_ma : 1.0);

   // 16  High-Low como % del precio
   feat[0][16] = (float)((h0 - l0) / c0 * 100.0);

   // 17  Posición del cierre en el rango [0,1]
   feat[0][17] = (float)(hl_rango > 0 ? (c0 - l0) / hl_rango : 0.5);

   // 18  Ratio del cuerpo [-1,1]
   feat[0][18] = (float)(hl_rango > 0 ? (c0 - o0) / hl_rango : 0.0);

   // 19  Williams %R normalizado [0,1]
   feat[0][19] = (float)(willr_rng > 0 ? (c0 - min_l) / willr_rng : 0.5);

   return true;
}

//====================================================================
//  ABRIR UNA OPERACIÓN CON SL/TP BASADOS EN ATR
//====================================================================

void AbrirOperacion(ENUM_ORDER_TYPE tipo, double pred)
{
   double atr_arr[];
   ArraySetAsSeries(atr_arr, true);
   if(CopyBuffer(g_h_atr, 0, 1, 1, atr_arr) < 1) return;
   double atr_val = atr_arr[0];
   if(atr_val <= 0) return;

   double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double price = (tipo == ORDER_TYPE_BUY) ? ask : bid;
   double sl_d  = atr_val * InpSL_ATR_Mult;
   double tp_d  = atr_val * InpTP_ATR_Mult;
   double sl, tp;

   if(tipo == ORDER_TYPE_BUY)
   { sl = price - sl_d;  tp = price + tp_d; }
   else
   { sl = price + sl_d;  tp = price - tp_d; }

   double lots = CalcularLotes(sl_d);
   if(lots <= 0) return;

   string dir = (tipo == ORDER_TYPE_BUY) ? "COMPRA 📈" : "VENTA  📉";
   Print("🚀 ", dir,
         " | Pred=", DoubleToString(pred, 4), "%",
         " | Precio=", DoubleToString(price, _Digits),
         " | SL=",     DoubleToString(sl,    _Digits),
         " | TP=",     DoubleToString(tp,    _Digits),
         " | Lotes=",  DoubleToString(lots,  2));

   bool ok;
   if(tipo == ORDER_TYPE_BUY)
      ok = g_trade.Buy (lots, _Symbol, price, sl, tp, "HybridAI-BUY");
   else
      ok = g_trade.Sell(lots, _Symbol, price, sl, tp, "HybridAI-SELL");

   if(!ok)
      Print("❌ Error al abrir: ", g_trade.ResultRetcodeDescription());
}

//====================================================================
//  CALCULAR TAMAÑO DE LOTE POR RIESGO FIJO (% DEL BALANCE)
//====================================================================

double CalcularLotes(double distancia_sl)
{
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double riesgo_usd  = balance * InpRiesgoPct / 100.0;

   double tick_val    = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double vol_min     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vol_max     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vol_step    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(tick_val <= 0 || tick_size <= 0 || distancia_sl <= 0) return vol_min;

   double ticks_sl    = distancia_sl / tick_size;
   double riesgo_lot  = ticks_sl * tick_val;

   if(riesgo_lot <= 0) return vol_min;

   double lotes = riesgo_usd / riesgo_lot;
   lotes = MathFloor(lotes / vol_step) * vol_step;
   lotes = MathMax(vol_min, MathMin(vol_max, lotes));

   return lotes;
}

//====================================================================
//  CONTAR POSICIONES ABIERTAS DE ESTE EA EN EL SÍMBOLO ACTUAL
//====================================================================

int ContarPosiciones(ENUM_POSITION_TYPE tipo)
{
   int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) == _Symbol &&
         (long)PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == tipo)
         n++;
   }
   return n;
}
//+------------------------------------------------------------------+
