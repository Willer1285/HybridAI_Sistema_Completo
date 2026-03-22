//+------------------------------------------------------------------+
//|                                        HybridAI_EA.mq5  v4.0    |
//|          Sistema de Trading con Inteligencia Artificial           |
//|          Modelo: ExtraTrees Clasificador + ONNX | M15             |
//|          Instrumento: XAUUSD (Gold)                               |
//|                                                                  |
//|  v4.0: Multi-timeframe (M15+H1+H4+D1), clasificación 3 clases,  |
//|        features de estructura de mercado, 40 features total       |
//+------------------------------------------------------------------+
#property copyright   "HybridAI Trading System 2026 v4.0"
#property description "EA con modelo de IA multi-TF para XAUUSD"
#property version     "4.00"
#property strict
#property tester_file "hybrid_ai_model.onnx"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//====================================================================
//  PARÁMETROS DE ENTRADA
//====================================================================

input group "==== MODELO IA ==============================="
input string   InpModelFile      = "hybrid_ai_model.onnx";

input group "==== UMBRALES DE SEÑAL ========================"
input double   InpUmbralBuy      =  0.45;   // Probabilidad mínima para COMPRA (0.0-1.0)
input double   InpUmbralSell     =  0.45;   // Probabilidad mínima para VENTA  (0.0-1.0)

input group "==== GESTIÓN DE RIESGO ======================="
input double   InpRiesgoPct      =  1.0;    // Riesgo por operación (% del balance)
input double   InpSL_ATR_Mult    =  2.0;    // Stop Loss  = ATR x este multiplicador
input double   InpTP_ATR_Mult    =  3.5;    // Take Profit= ATR x este multiplicador
input int      InpMaxTrades      =  1;      // Máximo de trades abiertos a la vez
input double   InpMaxLotes       =  5.0;    // Máximo de lotes por operación

input group "==== TRAILING STOP ==========================="
input bool     InpTrailingStop   = true;     // Activar trailing stop
input double   InpTrailATR_Mult  =  1.5;    // Trailing Stop = ATR x multiplicador
input double   InpBreakeven_ATR  =  1.0;    // Breakeven cuando ganancia > ATR x mult

input group "==== FILTROS ================================="
input bool     InpFiltroHora     = false;   // Activar filtro de hora
input int      InpHoraInicio     =  7;      // Hora inicio (GMT)
input int      InpHoraFin        = 20;      // Hora fin   (GMT)
input bool     InpFiltroSpread   = true;    // Activar filtro de spread
input double   InpMaxSpreadATR   =  0.10;   // Spread maximo como % del ATR

input group "==== AVANZADO ================================"
input bool     InpCerrarContraria = true;   // Cerrar posición contraria antes de abrir

input group "==== IDENTIFICACIÓN =========================="
input int      InpMagicNumber    = 246810;  // Número mágico del EA

//====================================================================
//  VARIABLES GLOBALES
//====================================================================

long      g_onnx     = INVALID_HANDLE;
CTrade    g_trade;

// Handles M15
int       g_h_atr    = INVALID_HANDLE;
int       g_h_rsi    = INVALID_HANDLE;
int       g_h_macd   = INVALID_HANDLE;
int       g_h_bb     = INVALID_HANDLE;
int       g_h_ema9   = INVALID_HANDLE;
int       g_h_ema21  = INVALID_HANDLE;
int       g_h_ema50  = INVALID_HANDLE;

// Handles H1
int       g_h1_rsi   = INVALID_HANDLE;
int       g_h1_macd  = INVALID_HANDLE;
int       g_h1_ema21 = INVALID_HANDLE;
int       g_h1_ema50 = INVALID_HANDLE;
int       g_h1_atr   = INVALID_HANDLE;

// Handles H4
int       g_h4_rsi   = INVALID_HANDLE;
int       g_h4_macd  = INVALID_HANDLE;
int       g_h4_ema21 = INVALID_HANDLE;
int       g_h4_ema50 = INVALID_HANDLE;
int       g_h4_atr   = INVALID_HANDLE;

// Handles D1
int       g_d1_rsi   = INVALID_HANDLE;
int       g_d1_macd  = INVALID_HANDLE;
int       g_d1_ema21 = INVALID_HANDLE;
int       g_d1_ema50 = INVALID_HANDLE;
int       g_d1_atr   = INVALID_HANDLE;

#define N_FEAT    40      // 20 M15 + 5 H1 + 5 H4 + 5 D1 + 5 estructura
#define LOOKBACK  210     // Barras M15 necesarias (200 para estructura + margen)

//====================================================================
//  DETECCIÓN AUTOMÁTICA DE FILLING MODE
//====================================================================
ENUM_ORDER_TYPE_FILLING DetectFillingMode()
{
   long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}

//====================================================================
//  INICIALIZACIÓN
//====================================================================

int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(DetectFillingMode());

   //-- Cargar ONNX
   g_onnx = OnnxCreate(InpModelFile, ONNX_DEFAULT);
   if(g_onnx == INVALID_HANDLE)
   {
      Alert("HybridAI: No se pudo cargar ONNX: ", InpModelFile);
      return INIT_FAILED;
   }

   //-- Diagnosticar modelo ONNX
   long n_inputs  = OnnxGetInputCount(g_onnx);
   long n_outputs = OnnxGetOutputCount(g_onnx);
   Print("ONNX cargado: ", n_inputs, " inputs, ", n_outputs, " outputs");

   // Mostrar info de cada input/output
   for(long i = 0; i < n_inputs; i++)
   {
      OnnxTypeInfo type_info;
      OnnxGetInputTypeInfo(g_onnx, i, type_info);
      string name = OnnxGetInputName(g_onnx, i);
      Print("  Input ", i, ": name=", name,
            " type=", EnumToString(type_info.tensor.data_type));
   }
   for(long i = 0; i < n_outputs; i++)
   {
      OnnxTypeInfo type_info;
      OnnxGetOutputTypeInfo(g_onnx, i, type_info);
      string name = OnnxGetOutputName(g_onnx, i);
      Print("  Output ", i, ": name=", name,
            " type=", EnumToString(type_info.tensor.data_type));
   }

   //-- Configurar shapes
   long sh_in[]  = {1, N_FEAT};

   if(!OnnxSetInputShape(g_onnx, 0, sh_in))
   {
      Print("ERROR: Input shape {1,", N_FEAT, "} fallo. Codigo: ", GetLastError());
      // Intentar sin batch dimension
      long sh_in2[] = {N_FEAT};
      if(!OnnxSetInputShape(g_onnx, 0, sh_in2))
      {
         Alert("HybridAI: Input shape fallo con ambos formatos. Codigo: ", GetLastError());
         return INIT_FAILED;
      }
      Print("Input shape alternativo {", N_FEAT, "} aceptado");
   }
   else
      Print("Input shape {1, ", N_FEAT, "} OK");

   //-- Output 0: labels (int64)
   long sh_out_label[] = {1};
   if(!OnnxSetOutputShape(g_onnx, 0, sh_out_label))
   {
      Print("WARN: Output 0 shape {1} fallo (", GetLastError(), "), intentando sin shape...");
      // Algunos modelos no necesitan output shape explícito
   }
   else
      Print("Output 0 (label) shape {1} OK");

   //-- Output 1: probabilities (float, shape [1, 3])
   if(n_outputs > 1)
   {
      long sh_out_prob[] = {1, 3};
      if(!OnnxSetOutputShape(g_onnx, 1, sh_out_prob))
      {
         Print("WARN: Output 1 shape {1,3} fallo (", GetLastError(), ")");
      }
      else
         Print("Output 1 (prob) shape {1, 3} OK");
   }

   //-- Indicadores M15
   g_h_atr   = iATR (_Symbol, PERIOD_M15, 14);
   g_h_rsi   = iRSI (_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   g_h_macd  = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
   g_h_bb    = iBands(_Symbol, PERIOD_M15, 20, 0, 2.0, PRICE_CLOSE);
   g_h_ema9  = iMA  (_Symbol, PERIOD_M15,  9, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema21 = iMA  (_Symbol, PERIOD_M15, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema50 = iMA  (_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);

   //-- Indicadores H1
   g_h1_rsi   = iRSI (_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   g_h1_macd  = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
   g_h1_ema21 = iMA  (_Symbol, PERIOD_H1, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_h1_ema50 = iMA  (_Symbol, PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE);
   g_h1_atr   = iATR (_Symbol, PERIOD_H1, 14);

   //-- Indicadores H4
   g_h4_rsi   = iRSI (_Symbol, PERIOD_H4, 14, PRICE_CLOSE);
   g_h4_macd  = iMACD(_Symbol, PERIOD_H4, 12, 26, 9, PRICE_CLOSE);
   g_h4_ema21 = iMA  (_Symbol, PERIOD_H4, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_h4_ema50 = iMA  (_Symbol, PERIOD_H4, 50, 0, MODE_EMA, PRICE_CLOSE);
   g_h4_atr   = iATR (_Symbol, PERIOD_H4, 14);

   //-- Indicadores D1
   g_d1_rsi   = iRSI (_Symbol, PERIOD_D1, 14, PRICE_CLOSE);
   g_d1_macd  = iMACD(_Symbol, PERIOD_D1, 12, 26, 9, PRICE_CLOSE);
   g_d1_ema21 = iMA  (_Symbol, PERIOD_D1, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_d1_ema50 = iMA  (_Symbol, PERIOD_D1, 50, 0, MODE_EMA, PRICE_CLOSE);
   g_d1_atr   = iATR (_Symbol, PERIOD_D1, 14);

   //-- Validar handles
   if(g_h_atr == INVALID_HANDLE || g_h_rsi == INVALID_HANDLE ||
      g_h1_rsi == INVALID_HANDLE || g_h4_rsi == INVALID_HANDLE ||
      g_d1_rsi == INVALID_HANDLE)
   {
      Alert("HybridAI: Error creando indicadores");
      return INIT_FAILED;
   }

   Print("================================================================");
   Print("    HybridAI EA v4.0 - XAUUSD Multi-Timeframe");
   Print("================================================================");
   Print("  Simbolo  : ", _Symbol);
   Print("  Modelo   : ", InpModelFile, " (", N_FEAT, " features, clasificacion)");
   Print("  Riesgo   : ", InpRiesgoPct, "%  SL=", InpSL_ATR_Mult, "xATR  TP=", InpTP_ATR_Mult, "xATR");
   Print("  Umbrales : BUY>=", InpUmbralBuy, "  SELL>=", InpUmbralSell);
   Print("  Max lotes: ", InpMaxLotes);
   Print("  Clases   : 0=NEUTRAL, 1=BUY, 2=SELL");
   Print("================================================================");

   return INIT_SUCCEEDED;
}

//====================================================================
//  LIBERACIÓN DE RECURSOS
//====================================================================

void OnDeinit(const int reason)
{
   if(g_onnx != INVALID_HANDLE) OnnxRelease(g_onnx);

   // Liberar todos los handles
   int handles[] = {
      g_h_atr, g_h_rsi, g_h_macd, g_h_bb, g_h_ema9, g_h_ema21, g_h_ema50,
      g_h1_rsi, g_h1_macd, g_h1_ema21, g_h1_ema50, g_h1_atr,
      g_h4_rsi, g_h4_macd, g_h4_ema21, g_h4_ema50, g_h4_atr,
      g_d1_rsi, g_d1_macd, g_d1_ema21, g_d1_ema50, g_d1_atr
   };
   for(int i = 0; i < ArraySize(handles); i++)
      if(handles[i] != INVALID_HANDLE) IndicatorRelease(handles[i]);

   Print("HybridAI v4.0 detenido.");
}

//====================================================================
//  TICK PRINCIPAL
//====================================================================

void OnTick()
{
   static datetime ultima_barra = 0;
   datetime barra_actual = iTime(_Symbol, PERIOD_M15, 0);
   if(barra_actual == ultima_barra) return;
   ultima_barra = barra_actual;

   if(Bars(_Symbol, PERIOD_M15) < LOOKBACK) return;

   //-- Filtro de hora (opcional)
   if(InpFiltroHora)
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < InpHoraInicio || dt.hour >= InpHoraFin) return;
   }

   if(InpTrailingStop)
      GestionarTrailingStop();

   // Filtro de spread
   double atr_check[];
   ArraySetAsSeries(atr_check, true);
   if(CopyBuffer(g_h_atr, 0, 1, 1, atr_check) < 1) return;
   double atr_current = atr_check[0];

   if(InpFiltroSpread && atr_current > 0)
   {
      double spread_val = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
      if(spread_val > atr_current * InpMaxSpreadATR) return;
   }

   // Calcular features
   matrixf features(1, N_FEAT);
   if(!CalcularFeatures(features)) return;

   // Inferencia ONNX - clasificador con 2 outputs (label + probabilities)
   vectorf out_label(1);       // Output 0: clase predicha (int64 -> cast a float)
   matrixf out_prob(1, 3);     // Output 1: probabilidades [NEUTRAL, BUY, SELL]

   bool run_ok = OnnxRun(g_onnx, ONNX_DEFAULT, features, out_label, out_prob);

   int clase = 0;  // default NEUTRAL
   float prob_neutral = 0.0f, prob_buy = 0.0f, prob_sell = 0.0f;

   if(!run_ok)
   {
      // Intentar con solo 1 output (label)
      vectorf out_single(1);
      if(!OnnxRun(g_onnx, ONNX_DEFAULT, features, out_single))
      {
         static int err_count = 0;
         if(++err_count <= 3)
            Print("OnnxRun fallo | Error: ", GetLastError());
         return;
      }
      clase = (int)out_single[0];
      // Sin probabilidades, asignar 1.0 a la clase predicha
      if(clase == 0) prob_neutral = 1.0f;
      else if(clase == 1) prob_buy = 1.0f;
      else prob_sell = 1.0f;
   }
   else
   {
      prob_neutral = out_prob[0][0];
      prob_buy     = out_prob[0][1];
      prob_sell    = out_prob[0][2];

      // Determinar clase usando probabilidades
      float max_prob = prob_neutral;
      clase = 0;
      if(prob_buy  > max_prob) { max_prob = prob_buy;  clase = 1; }
      if(prob_sell > max_prob) { max_prob = prob_sell; clase = 2; }
   }

   // Log cada 10 barras
   static int cnt = 0;
   if(++cnt % 10 == 0)
   {
      string clase_str = (clase == 1) ? "BUY" : (clase == 2) ? "SELL" : "NEUTRAL";
      Print(_Symbol, " | ", TimeToString(barra_actual, TIME_DATE|TIME_MINUTES),
            " | Clase: ", clase_str,
            " | P[N]=", DoubleToString(prob_neutral, 3),
            " P[B]=", DoubleToString(prob_buy, 3),
            " P[S]=", DoubleToString(prob_sell, 3));
   }

   int n_buy  = ContarPosiciones(POSITION_TYPE_BUY);
   int n_sell = ContarPosiciones(POSITION_TYPE_SELL);
   int n_tot  = n_buy + n_sell;

   // SEÑAL DE COMPRA: clase BUY + probabilidad >= umbral
   if(clase == 1 && prob_buy >= (float)InpUmbralBuy)
   {
      if(InpCerrarContraria && n_sell > 0)
         CerrarPosiciones(POSITION_TYPE_SELL);
      if(n_buy == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_BUY, clase);
   }
   // SEÑAL DE VENTA: clase SELL + probabilidad >= umbral
   else if(clase == 2 && prob_sell >= (float)InpUmbralSell)
   {
      if(InpCerrarContraria && n_buy > 0)
         CerrarPosiciones(POSITION_TYPE_BUY);
      if(n_sell == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_SELL, clase);
   }
   // NEUTRAL o probabilidad por debajo del umbral → no operar
}

//====================================================================
//  CALCULAR 40 FEATURES v4.0
//  0-19:   M15 técnicos
//  20-24:  H1 (rsi, macd_hist, ema_trend, atr_pct, ret3)
//  25-29:  H4 (rsi, macd_hist, ema_trend, atr_pct, ret3)
//  30-34:  D1 (rsi, macd_hist, ema_trend, atr_pct, ret3)
//  35-39:  Estructura (pos_range, hh_ratio, dist_max, dist_min, hour_sin)
//====================================================================

bool CalcularFeatures(matrixf &feat)
{
   const int LB = 205;

   //-- OHLCV M15
   double close[], high[], low[], open_arr[];
   long   vol[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(high, true);
   ArraySetAsSeries(open_arr, true); ArraySetAsSeries(low, true);
   ArraySetAsSeries(vol, true);

   if(CopyClose     (_Symbol, PERIOD_M15, 1, LB, close)    < LB) return false;
   if(CopyHigh      (_Symbol, PERIOD_M15, 1, LB, high)     < LB) return false;
   if(CopyLow       (_Symbol, PERIOD_M15, 1, LB, low)      < LB) return false;
   if(CopyOpen      (_Symbol, PERIOD_M15, 1, LB, open_arr) < LB) return false;
   if(CopyTickVolume(_Symbol, PERIOD_M15, 1, LB, vol)      < LB) return false;

   //-- Indicadores M15
   double atr[], rsi[], macd_m[], macd_s[];
   double bb_u[], bb_mid[], bb_l[];
   double ema9a[], ema21a[], ema50a[];
   ArraySetAsSeries(atr, true); ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(macd_m, true); ArraySetAsSeries(macd_s, true);
   ArraySetAsSeries(bb_u, true); ArraySetAsSeries(bb_mid, true); ArraySetAsSeries(bb_l, true);
   ArraySetAsSeries(ema9a, true); ArraySetAsSeries(ema21a, true); ArraySetAsSeries(ema50a, true);

   if(CopyBuffer(g_h_atr,   0, 1, LB, atr)    < LB) return false;
   if(CopyBuffer(g_h_rsi,   0, 1, LB, rsi)    < LB) return false;
   if(CopyBuffer(g_h_macd,  0, 1, LB, macd_m) < LB) return false;
   if(CopyBuffer(g_h_macd,  1, 1, LB, macd_s) < LB) return false;
   if(CopyBuffer(g_h_bb,    1, 1, LB, bb_u)   < LB) return false;
   if(CopyBuffer(g_h_bb,    0, 1, LB, bb_mid) < LB) return false;
   if(CopyBuffer(g_h_bb,    2, 1, LB, bb_l)   < LB) return false;
   if(CopyBuffer(g_h_ema9,  0, 1, LB, ema9a)  < LB) return false;
   if(CopyBuffer(g_h_ema21, 0, 1, LB, ema21a) < LB) return false;
   if(CopyBuffer(g_h_ema50, 0, 1, LB, ema50a) < LB) return false;

   double c0   = close[0];
   double h0   = high[0];
   double l0   = low[0];
   double o0   = open_arr[0];
   double atr0 = atr[0];
   if(atr0 <= 0.0 || c0 <= 0.0) return false;

   double bb_ancho = bb_u[0] - bb_l[0];
   double hl_rango = h0 - l0;

   double vol_sum = 0;
   for(int i = 0; i < 20; i++) vol_sum += (double)vol[i];
   double vol_ma = vol_sum / 20.0;

   double max_h = high[0], min_l = low[0];
   for(int i = 1; i < 14; i++)
   {
      if(high[i] > max_h) max_h = high[i];
      if(low[i]  < min_l) min_l = low[i];
   }
   double willr_rng = max_h - min_l;

   // Features 0-19: M15 (idéntico a Python)
   feat[0][0]  = (float)(rsi[0] / 100.0);
   feat[0][1]  = (float)(macd_m[0] / atr0);
   feat[0][2]  = (float)(macd_s[0] / atr0);
   feat[0][3]  = (float)((macd_m[0] - macd_s[0]) / atr0);
   feat[0][4]  = (float)(atr0 / c0);
   feat[0][5]  = (float)(bb_ancho > 0 ? (c0 - bb_l[0]) / bb_ancho : 0.5);
   feat[0][6]  = (float)(bb_mid[0] > 0 ? bb_ancho / bb_mid[0] : 0.0);
   feat[0][7]  = (float)(ema9a[0]  > 0 ? (c0 - ema9a[0])  / ema9a[0]  * 100.0 : 0.0);
   feat[0][8]  = (float)(ema21a[0] > 0 ? (c0 - ema21a[0]) / ema21a[0] * 100.0 : 0.0);
   feat[0][9]  = (float)(ema50a[0] > 0 ? (c0 - ema50a[0]) / ema50a[0] * 100.0 : 0.0);
   feat[0][10] = (float)(close[1]  > 0 ? (c0 - close[1])  / close[1]  * 100.0 : 0.0);
   feat[0][11] = (float)(close[3]  > 0 ? (c0 - close[3])  / close[3]  * 100.0 : 0.0);
   feat[0][12] = (float)(close[5]  > 0 ? (c0 - close[5])  / close[5]  * 100.0 : 0.0);
   feat[0][13] = (float)(close[10] > 0 ? (c0 - close[10]) / close[10] * 100.0 : 0.0);
   feat[0][14] = (float)(close[20] > 0 ? (c0 - close[20]) / close[20] * 100.0 : 0.0);
   feat[0][15] = (float)(vol_ma > 0 ? (double)vol[0] / vol_ma : 1.0);
   feat[0][16] = (float)((h0 - l0) / c0 * 100.0);
   feat[0][17] = (float)(hl_rango > 0 ? (c0 - l0) / hl_rango : 0.5);
   feat[0][18] = (float)(hl_rango > 0 ? (c0 - o0) / hl_rango : 0.0);
   feat[0][19] = (float)(willr_rng > 0 ? (c0 - min_l) / willr_rng : 0.5);

   //-- Features 20-34: HTF (H1, H4, D1)
   if(!CalcularHTF(feat, 20, g_h1_rsi, g_h1_macd, g_h1_ema21, g_h1_ema50, g_h1_atr, PERIOD_H1)) return false;
   if(!CalcularHTF(feat, 25, g_h4_rsi, g_h4_macd, g_h4_ema21, g_h4_ema50, g_h4_atr, PERIOD_H4)) return false;
   if(!CalcularHTF(feat, 30, g_d1_rsi, g_d1_macd, g_d1_ema21, g_d1_ema50, g_d1_atr, PERIOD_D1)) return false;

   //-- Features 35-39: Estructura
   // 35: Posición en rango de 96 barras (1 día)
   double max96 = high[0], min96 = low[0];
   for(int i = 1; i < 96; i++)
   {
      if(high[i] > max96) max96 = high[i];
      if(low[i]  < min96) min96 = low[i];
   }
   double rng96 = max96 - min96;
   feat[0][35] = (float)(rng96 > 0 ? (c0 - min96) / rng96 : 0.5);

   // 36: Higher-highs ratio (20 barras)
   int hh_count = 0;
   for(int i = 0; i < 19; i++)
      if(high[i] > high[i+1]) hh_count++;
   feat[0][36] = (float)(hh_count / 19.0);

   // 37: Distancia al máximo de 200 barras / ATR
   double max200 = high[0];
   for(int i = 1; i < 200; i++)
      if(high[i] > max200) max200 = high[i];
   feat[0][37] = (float)((max200 - c0) / atr0);

   // 38: Distancia al mínimo de 200 barras / ATR
   double min200 = low[0];
   for(int i = 1; i < 200; i++)
      if(low[i] < min200) min200 = low[i];
   feat[0][38] = (float)((c0 - min200) / atr0);

   // 39: Hora cíclica (sin)
   MqlDateTime bar_time;
   TimeToStruct(iTime(_Symbol, PERIOD_M15, 1), bar_time);
   double hora = (double)bar_time.hour + (double)bar_time.min / 60.0;
   feat[0][39] = (float)(MathSin(2.0 * M_PI * hora / 24.0));

   return true;
}

//====================================================================
//  CALCULAR 5 FEATURES DE UN HTF
//====================================================================

bool CalcularHTF(matrixf &feat, int offset,
                 int h_rsi, int h_macd, int h_ema21, int h_ema50, int h_atr,
                 ENUM_TIMEFRAMES tf)
{
   double rsi_v[], macd_m[], macd_s[], ema21_v[], ema50_v[], atr_v[];
   double close_htf[];
   ArraySetAsSeries(rsi_v, true); ArraySetAsSeries(macd_m, true);
   ArraySetAsSeries(macd_s, true); ArraySetAsSeries(ema21_v, true);
   ArraySetAsSeries(ema50_v, true); ArraySetAsSeries(atr_v, true);
   ArraySetAsSeries(close_htf, true);

   if(CopyBuffer(h_rsi,  0, 1, 5, rsi_v)   < 1) return false;
   if(CopyBuffer(h_macd, 0, 1, 5, macd_m)  < 1) return false;
   if(CopyBuffer(h_macd, 1, 1, 5, macd_s)  < 1) return false;
   if(CopyBuffer(h_ema21,0, 1, 5, ema21_v) < 1) return false;
   if(CopyBuffer(h_ema50,0, 1, 5, ema50_v) < 1) return false;
   if(CopyBuffer(h_atr,  0, 1, 5, atr_v)   < 1) return false;
   if(CopyClose(_Symbol, tf, 1, 5, close_htf) < 4) return false;

   double atr_htf = atr_v[0];
   if(atr_htf <= 0) atr_htf = 1e-10;

   // 0: RSI normalizado
   feat[0][offset + 0] = (float)(rsi_v[0] / 100.0);
   // 1: MACD histograma / ATR
   feat[0][offset + 1] = (float)((macd_m[0] - macd_s[0]) / atr_htf);
   // 2: Tendencia EMA = (EMA21 - EMA50) / ATR
   feat[0][offset + 2] = (float)((ema21_v[0] - ema50_v[0]) / atr_htf);
   // 3: ATR como % del precio
   feat[0][offset + 3] = (float)(close_htf[0] > 0 ? atr_htf / close_htf[0] : 0.0);
   // 4: Retorno 3 barras del HTF
   feat[0][offset + 4] = (float)(close_htf[3] > 0 ? (close_htf[0] - close_htf[3]) / close_htf[3] * 100.0 : 0.0);

   return true;
}

//====================================================================
//  ABRIR OPERACIÓN
//====================================================================

void AbrirOperacion(ENUM_ORDER_TYPE tipo, int clase)
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

   double tick = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick > 0)
   {
      sl = MathRound(sl / tick) * tick;
      tp = MathRound(tp / tick) * tick;
   }

   double lots = CalcularLotes(sl_d);
   if(lots <= 0) return;

   string dir = (tipo == ORDER_TYPE_BUY) ? "COMPRA" : "VENTA";
   Print(">> ", dir,
         " | Clase=", clase,
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
      Print("Error al abrir: ", g_trade.ResultRetcodeDescription());
}

//====================================================================
//  CALCULAR LOTES
//====================================================================

double CalcularLotes(double distancia_sl)
{
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double riesgo_usd = balance * InpRiesgoPct / 100.0;
   double tick_val   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double vol_min    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vol_max    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vol_step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(tick_val <= 0 || tick_size <= 0 || distancia_sl <= 0) return vol_min;

   double ticks_sl   = distancia_sl / tick_size;
   double riesgo_lot = ticks_sl * tick_val;
   if(riesgo_lot <= 0) return vol_min;

   double lotes = riesgo_usd / riesgo_lot;
   lotes = MathFloor(lotes / vol_step) * vol_step;
   double cap = MathMin(vol_max, InpMaxLotes);
   lotes = MathMax(vol_min, MathMin(cap, lotes));
   return lotes;
}

//====================================================================
//  TRAILING STOP
//====================================================================

void GestionarTrailingStop()
{
   double atr_arr[];
   ArraySetAsSeries(atr_arr, true);
   if(CopyBuffer(g_h_atr, 0, 1, 1, atr_arr) < 1) return;
   double atr_val = atr_arr[0];
   if(atr_val <= 0) return;

   double trail_dist = atr_val * InpTrailATR_Mult;
   double be_dist    = atr_val * InpBreakeven_ATR;
   double tick       = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick <= 0) tick = _Point;

   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

      double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_sl   = PositionGetDouble(POSITION_SL);
      double pos_tp   = PositionGetDouble(POSITION_TP);
      long   pos_type = PositionGetInteger(POSITION_TYPE);
      ulong  pos_ticket = PositionGetInteger(POSITION_TICKET);

      if(pos_type == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit_dist = bid - pos_open;
         if(profit_dist >= be_dist && pos_sl < pos_open)
         {
            double new_sl = MathRound(pos_open / tick) * tick;
            if(new_sl > pos_sl) g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
         else if(profit_dist >= trail_dist)
         {
            double new_sl = MathRound((bid - trail_dist) / tick) * tick;
            if(new_sl > pos_sl) g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
      }
      else if(pos_type == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit_dist = pos_open - ask;
         if(profit_dist >= be_dist && (pos_sl > pos_open || pos_sl == 0))
         {
            double new_sl = MathRound(pos_open / tick) * tick;
            if(new_sl < pos_sl || pos_sl == 0) g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
         else if(profit_dist >= trail_dist)
         {
            double new_sl = MathRound((ask + trail_dist) / tick) * tick;
            if(new_sl < pos_sl || pos_sl == 0) g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
      }
   }
}

//====================================================================
//  CERRAR POSICIONES
//====================================================================

void CerrarPosiciones(ENUM_POSITION_TYPE tipo)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != tipo) continue;
      ulong ticket = PositionGetInteger(POSITION_TICKET);
      g_trade.PositionClose(ticket);
      Print("Cerrada posicion contraria #", ticket);
   }
}

//====================================================================
//  CONTAR POSICIONES
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
