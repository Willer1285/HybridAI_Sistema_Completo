//+------------------------------------------------------------------+
//|                                        HybridAI_EA.mq5  v3.0    |
//|          Sistema de Trading con Inteligencia Artificial           |
//|          Modelo: ExtraTrees + ONNX | Timeframe: M15              |
//|          Instrumento: XAUUSD (Gold) solamente                    |
//|                                                                  |
//|  v3.0: Features de régimen de volatilidad, fuerza de tendencia,  |
//|        hora cíclica. Cap de lotes. Sin multi-symbol.             |
//+------------------------------------------------------------------+
#property copyright   "HybridAI Trading System 2026 v3.0"
#property description "EA con modelo de IA (ONNX) para XAUUSD"
#property version     "3.00"
#property strict
#property tester_file "hybrid_ai_model.onnx"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//====================================================================
//  PARÁMETROS DE ENTRADA
//====================================================================

input group "==== MODELO IA ==============================="
input string   InpModelFile      = "hybrid_ai_model.onnx";
//  ^-- Nombre del archivo ONNX en la carpeta MQL5\Files\

input group "==== UMBRALES DE SEÑAL ======================="
input double   InpUmbralCompra   =  0.15;   // Comprar si predicción > este valor (%)
input double   InpUmbralVenta    = -0.15;   // Vender  si predicción < este valor (%)

input group "==== GESTIÓN DE RIESGO ======================="
input double   InpRiesgoPct      =  1.0;    // Riesgo por operación (% del balance)
input double   InpSL_ATR_Mult    =  2.0;    // Stop Loss  = ATR x este multiplicador
input double   InpTP_ATR_Mult    =  3.5;    // Take Profit= ATR x este multiplicador
input int      InpMaxTrades      =  1;      // Máximo de trades abiertos a la vez
input double   InpMaxLotes       =  5.0;    // Máximo de lotes por operación

input group "==== TRAILING STOP ==========================="
input bool     InpTrailingStop   = true;     // Activar trailing stop
input double   InpTrailATR_Mult  =  1.5;    // Trailing Stop = ATR x multiplicador
input double   InpBreakeven_ATR  =  1.0;    // Mover SL a breakeven cuando ganancia > ATR x mult

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

int       g_h_atr    = INVALID_HANDLE;
int       g_h_rsi    = INVALID_HANDLE;
int       g_h_macd   = INVALID_HANDLE;
int       g_h_bb     = INVALID_HANDLE;
int       g_h_ema9   = INVALID_HANDLE;
int       g_h_ema21  = INVALID_HANDLE;
int       g_h_ema50  = INVALID_HANDLE;

#define N_FEAT    24      // 20 técnicos + vol_regime + trend_str + hour_sin + hour_cos (v3.0)
#define LOOKBACK  65      // Barras mínimas para calentar indicadores (50 para ATR SMA50)

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
   //-- Configurar objeto de trading
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(DetectFillingMode());

   //-- Cargar el modelo ONNX
   g_onnx = OnnxCreate(InpModelFile, ONNX_DEFAULT);

   if(g_onnx == INVALID_HANDLE)
   {
      Alert("HybridAI: No se pudo cargar ONNX: ", InpModelFile,
            "\n\nVerifica que copiaste el archivo a:\n",
            "MT5 -> File -> Open Data Folder -> MQL5 -> Files");
      return INIT_FAILED;
   }

   //-- Configurar shapes de entrada/salida del modelo ONNX
   long sh_in[]  = {1, N_FEAT};   // [batch=1, features=24]
   long sh_out[] = {1, 1};        // [batch=1, outputs=1]

   if(!OnnxSetInputShape(g_onnx, 0, sh_in))
   {
      Alert("HybridAI: Error configurando input shape. Código: ", GetLastError());
      return INIT_FAILED;
   }
   if(!OnnxSetOutputShape(g_onnx, 0, sh_out))
   {
      Alert("HybridAI: Error configurando output shape. Código: ", GetLastError());
      return INIT_FAILED;
   }

   //-- Crear handles de indicadores técnicos
   g_h_atr  = iATR (_Symbol, PERIOD_CURRENT, 14);
   g_h_rsi  = iRSI (_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   g_h_macd = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   g_h_bb   = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
   g_h_ema9  = iMA  (_Symbol, PERIOD_CURRENT,  9, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema21 = iMA  (_Symbol, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_h_ema50 = iMA  (_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);

   //-- Validar handles
   if(g_h_atr  == INVALID_HANDLE || g_h_rsi  == INVALID_HANDLE ||
      g_h_macd == INVALID_HANDLE || g_h_bb   == INVALID_HANDLE ||
      g_h_ema9  == INVALID_HANDLE || g_h_ema21 == INVALID_HANDLE ||
      g_h_ema50 == INVALID_HANDLE)
   {
      Alert("HybridAI: Error al crear indicadores. Código: ", GetLastError());
      return INIT_FAILED;
   }

   //-- Log de inicio
   Print("================================================================");
   Print("    HybridAI EA v3.0 - XAUUSD Only");
   Print("================================================================");
   Print("  Simbolo  : ", _Symbol);
   Print("  Timeframe: ", EnumToString(PERIOD_CURRENT));
   Print("  Modelo   : ", InpModelFile, " -> CARGADO (", N_FEAT, " features)");
   Print("  Umbral   : Compra >", InpUmbralCompra, "%  |  Venta <", InpUmbralVenta, "%");
   Print("  Riesgo   : ", InpRiesgoPct, "%  |  SL=", InpSL_ATR_Mult, "xATR  |  TP=", InpTP_ATR_Mult, "xATR");
   Print("  Max lotes: ", InpMaxLotes);
   Print("  Trailing : ", InpTrailingStop ? "ON" : "OFF",
         "  (", InpTrailATR_Mult, "xATR, BE=", InpBreakeven_ATR, "xATR)");
   Print("  Spread   : ", InpFiltroSpread ? "Filtro ON" : "Sin filtro",
         "  (max ", InpMaxSpreadATR*100, "% ATR)");
   Print("  Filling  : ", EnumToString(DetectFillingMode()));
   Print("================================================================");

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

   Print("HybridAI v3.0 detenido.");
}

//====================================================================
//  TICK PRINCIPAL
//====================================================================

void OnTick()
{
   //-- Solo actuar en una nueva barra completada
   static datetime ultima_barra = 0;
   datetime barra_actual = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(barra_actual == ultima_barra) return;
   ultima_barra = barra_actual;

   //-- Esperar suficientes barras para indicadores
   if(Bars(_Symbol, PERIOD_CURRENT) < LOOKBACK) return;

   //-- Filtro de hora (opcional)
   if(InpFiltroHora)
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.hour < InpHoraInicio || dt.hour >= InpHoraFin) return;
   }

   //-- Gestionar trailing stop de posiciones abiertas
   if(InpTrailingStop)
      GestionarTrailingStop();

   //-- Obtener ATR para filtro de spread
   double atr_check[];
   ArraySetAsSeries(atr_check, true);
   if(CopyBuffer(g_h_atr, 0, 1, 1, atr_check) < 1) return;
   double atr_current = atr_check[0];

   //-- Filtro de spread
   if(InpFiltroSpread && atr_current > 0)
   {
      double spread_val = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
      if(spread_val > atr_current * InpMaxSpreadATR) return;
   }

   //-- Calcular los 24 features v3.0
   matrixf features(1, N_FEAT);
   if(!CalcularFeatures(features)) return;

   //-- Ejecutar inferencia ONNX
   vectorf salida(1);
   if(!OnnxRun(g_onnx, ONNX_DEFAULT, features, salida))
   {
      Print("OnnxRun fallo en ", TimeToString(barra_actual), " | Error: ", GetLastError());
      return;
   }

   double prediccion = (double)salida[0];

   //-- Log cada 10 barras
   static int cnt = 0;
   if(++cnt % 10 == 0)
      Print(_Symbol, " | ", TimeToString(barra_actual, TIME_DATE|TIME_MINUTES),
            " | Prediccion IA: ", DoubleToString(prediccion, 4), "%");

   //-- Contar posiciones abiertas
   int n_buy  = ContarPosiciones(POSITION_TYPE_BUY);
   int n_sell = ContarPosiciones(POSITION_TYPE_SELL);
   int n_tot  = n_buy + n_sell;

   //-- SEÑAL DE COMPRA
   if(prediccion > InpUmbralCompra)
   {
      // Cerrar posición contraria si existe
      if(InpCerrarContraria && n_sell > 0)
         CerrarPosiciones(POSITION_TYPE_SELL);

      if(n_buy == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_BUY, prediccion);
   }
   //-- SEÑAL DE VENTA
   else if(prediccion < InpUmbralVenta)
   {
      if(InpCerrarContraria && n_buy > 0)
         CerrarPosiciones(POSITION_TYPE_BUY);

      if(n_sell == 0 && n_tot < InpMaxTrades)
         AbrirOperacion(ORDER_TYPE_SELL, prediccion);
   }
}

//====================================================================
//  CALCULAR 24 FEATURES v3.0  <- DEBE SER IDÉNTICO AL PYTHON v3.0
//  0-19:  Indicadores técnicos base
//  20:    vol_regime  = ATR(14) / SMA(ATR(14), 50)
//  21:    trend_str   = (EMA21 - EMA50) / ATR(14)
//  22:    hour_sin    = sin(2π * hora / 24)
//  23:    hour_cos    = cos(2π * hora / 24)
//====================================================================

bool CalcularFeatures(matrixf &feat)
{
   const int LB = 55;   // barras históricas a pedir

   //-- Obtener OHLCV
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

   //-- Obtener buffers de indicadores
   double atr[], rsi[], macd_m[], macd_s[];
   double bb_u[], bb_mid[], bb_l[];
   double ema9a[], ema21a[], ema50a[];

   ArraySetAsSeries(atr,    true);  ArraySetAsSeries(rsi,    true);
   ArraySetAsSeries(macd_m, true);  ArraySetAsSeries(macd_s, true);
   ArraySetAsSeries(bb_u,   true);  ArraySetAsSeries(bb_mid, true);
   ArraySetAsSeries(bb_l,   true);
   ArraySetAsSeries(ema9a,  true);  ArraySetAsSeries(ema21a, true);
   ArraySetAsSeries(ema50a, true);

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

   //-- Valores de la barra más reciente completada (índice 0)
   double c0   = close[0];
   double h0   = high[0];
   double l0   = low[0];
   double o0   = open_arr[0];
   double atr0 = atr[0];

   if(atr0 <= 0.0 || c0 <= 0.0) return false;

   double bb_ancho = bb_u[0] - bb_l[0];
   double hl_rango = h0 - l0;

   //-- Volumen medio 20 barras
   double vol_sum = 0;
   for(int i = 0; i < 20; i++) vol_sum += (double)vol[i];
   double vol_ma = vol_sum / 20.0;

   //-- Williams %R(14)
   double max_h = high[0], min_l = low[0];
   for(int i = 1; i < 14; i++)
   {
      if(high[i] > max_h) max_h = high[i];
      if(low[i]  < min_l) min_l = low[i];
   }
   double willr_rng = max_h - min_l;

   //-- Features 0-19: técnicos (misma lógica que Python)

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

   //-- Features 20-23: v3.0 (régimen, tendencia, hora cíclica)

   // 20  vol_regime = ATR(14) / SMA(ATR(14), 50)
   //     Calculamos media simple de las últimas 50 barras de ATR
   double atr_sum50 = 0;
   for(int i = 0; i < 50; i++) atr_sum50 += atr[i];
   double atr_sma50 = atr_sum50 / 50.0;
   feat[0][20] = (float)(atr_sma50 > 0 ? atr0 / atr_sma50 : 1.0);

   // 21  trend_strength = (EMA21 - EMA50) / ATR(14)
   feat[0][21] = (float)((ema21a[0] - ema50a[0]) / atr0);

   // 22-23  Codificación cíclica de la hora
   MqlDateTime bar_time;
   TimeToStruct(iTime(_Symbol, PERIOD_CURRENT, 1), bar_time);
   double hora = (double)bar_time.hour + (double)bar_time.min / 60.0;
   feat[0][22] = (float)(MathSin(2.0 * M_PI * hora / 24.0));
   feat[0][23] = (float)(MathCos(2.0 * M_PI * hora / 24.0));

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

   // Normalizar SL/TP a tick size
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
      Print("Error al abrir: ", g_trade.ResultRetcodeDescription());
}

//====================================================================
//  CALCULAR TAMAÑO DE LOTE POR RIESGO FIJO (% DEL BALANCE)
//  Con cap máximo de InpMaxLotes (v3.0)
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

   // Cap de lotes v3.0
   double cap = MathMin(vol_max, InpMaxLotes);
   lotes = MathMax(vol_min, MathMin(cap, lotes));

   return lotes;
}

//====================================================================
//  TRAILING STOP - GESTIÓN DINÁMICA DE SL
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

      double pos_open  = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_sl    = PositionGetDouble(POSITION_SL);
      double pos_tp    = PositionGetDouble(POSITION_TP);
      long   pos_type  = PositionGetInteger(POSITION_TYPE);
      ulong  pos_ticket = PositionGetInteger(POSITION_TICKET);

      if(pos_type == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit_dist = bid - pos_open;

         // Breakeven: mover SL a entry cuando ganancia > be_dist
         if(profit_dist >= be_dist && pos_sl < pos_open)
         {
            double new_sl = MathRound(pos_open / tick) * tick;
            if(new_sl > pos_sl)
               g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
         // Trail: mover SL a bid - trail_dist
         else if(profit_dist >= trail_dist)
         {
            double new_sl = MathRound((bid - trail_dist) / tick) * tick;
            if(new_sl > pos_sl)
               g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
      }
      else if(pos_type == POSITION_TYPE_SELL)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit_dist = pos_open - ask;

         if(profit_dist >= be_dist && (pos_sl > pos_open || pos_sl == 0))
         {
            double new_sl = MathRound(pos_open / tick) * tick;
            if(new_sl < pos_sl || pos_sl == 0)
               g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
         else if(profit_dist >= trail_dist)
         {
            double new_sl = MathRound((ask + trail_dist) / tick) * tick;
            if(new_sl < pos_sl || pos_sl == 0)
               g_trade.PositionModify(pos_ticket, new_sl, pos_tp);
         }
      }
   }
}

//====================================================================
//  CERRAR POSICIONES DE UN TIPO (para cerrar contrarias)
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
