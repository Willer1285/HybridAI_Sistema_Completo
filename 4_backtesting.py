# =============================================================
#  PASO 4 - MOTOR DE BACKTESTING CON MÉTRICAS FINANCIERAS
#  Simula trading real con costos, spread, slippage y genera
#  métricas completas para los 4 activos objetivo.
#
#  Comando: python 4_backtesting.py
#
#  Requisitos: Haber ejecutado paso 3 (modelo ONNX entrenado)
# =============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

import onnxruntime as rt

# Reutilizar calcular_features del módulo de entrenamiento
import importlib.util
spec = importlib.util.spec_from_file_location("trainer", "3_entrenar_modelo.py")

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLOS       = ["xauusd", "eurusd", "gbpusd", "usdjpy"]
SYMBOL_IDS     = {s: i for i, s in enumerate(SIMBOLOS)}
CARPETA_DATOS  = "datos"
CARPETA_MODELO = "modelo"
N_FEATURES_BASE = 20
N_FEATURES      = 24  # 20 base + 4 one-hot
BARRAS_FUTURO   = 5

# Costos realistas por símbolo (en unidades de precio)
SPREAD_PIPS = {
    "xauusd": 0.30,   # $0.30 en gold
    "eurusd": 0.00010, # 1.0 pip
    "gbpusd": 0.00015, # 1.5 pips
    "usdjpy": 0.012,   # 1.2 pips
}
COMMISSION_PER_LOT = {
    "xauusd": 7.0,
    "eurusd": 7.0,
    "gbpusd": 7.0,
    "usdjpy": 7.0,
}
SLIPPAGE_PIPS = {
    "xauusd": 0.10,
    "eurusd": 0.00003,
    "gbpusd": 0.00004,
    "usdjpy": 0.004,
}

# Parámetros de trading por defecto
DEFAULT_PARAMS = {
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 3.0,
    "risk_pct": 1.0,
    "max_trades": 1,
}

INITIAL_BALANCE = 10000.0
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE FEATURES (duplicadas para independencia)   ║
# ╚══════════════════════════════════════════════════════════╝

def ema_calc(data, period):
    result = np.full(len(data), np.nan)
    first_valid = 0
    while first_valid < len(data) and np.isnan(data[first_valid]):
        first_valid += 1
    if first_valid + period > len(data):
        return result
    k = 2.0 / (period + 1)
    result[first_valid + period - 1] = np.mean(data[first_valid : first_valid + period])
    for i in range(first_valid + period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1.0 - k)
    return result

def sma_calc(data, period):
    result = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result

def atr_calc(h_, l_, c_, period):
    n = len(c_)
    tr = np.maximum(h_[1:] - l_[1:],
         np.maximum(np.abs(h_[1:] - c_[:-1]),
                    np.abs(l_[1:] - c_[:-1])))
    tr = np.concatenate([[np.nan], tr])
    result = np.full(n, np.nan)
    result[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result

def calcular_features_array(c, h, l, o, v):
    """Calcula los 20 features base a partir de arrays OHLCV."""
    n = len(c)
    feats = np.full((n, N_FEATURES_BASE), np.nan, dtype=np.float64)

    # RSI(14)
    rsi_p = 14
    delta = np.diff(c)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n, np.nan)
    al = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p])
    al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs  = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:rsi_p] = np.nan

    # MACD
    ema12 = ema_calc(c, 12)
    ema26 = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema_calc(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # ATR
    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    # Bollinger
    sma20 = sma_calc(c, 20)
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19:i+1], ddof=0)
    bb_up = sma20 + 2.0 * std20
    bb_lo = sma20 - 2.0 * std20
    bb_w  = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp  = np.where(sma20 == 0, np.nan, bb_w / sma20)

    # EMAs
    ema9  = ema_calc(c, 9)
    ema21 = ema_calc(c, 21)
    ema50 = ema_calc(c, 50)

    # Returns
    def ret(data, p):
        r = np.full(n, np.nan)
        r[p:] = (data[p:] - data[:-p]) / data[:-p] * 100.0
        return r

    # Volume
    vol_ma = sma_calc(v, 20)
    vol_rt = np.where(vol_ma == 0, 1.0, v / vol_ma)

    # Candle
    hl_r = np.where(c == 0, np.nan, (h - l) / c * 100.0)
    rng = h - l
    cl_p = np.where(rng == 0, 0.5, (c - l) / rng)
    bd_r = np.where(rng == 0, 0.0, (c - o) / rng)

    # Williams %R
    mxh = np.full(n, np.nan)
    mnl = np.full(n, np.nan)
    for i in range(13, n):
        mxh[i] = np.max(h[i-13:i+1])
        mnl[i] = np.min(l[i-13:i+1])
    wl_r = mxh - mnl
    will = np.where(wl_r == 0, 0.5, (c - mnl) / wl_r)

    feats[:, 0]  = rsi / 100.0
    feats[:, 1]  = macd_line / atr_s
    feats[:, 2]  = macd_sig  / atr_s
    feats[:, 3]  = macd_hist / atr_s
    feats[:, 4]  = np.where(c == 0, np.nan, atr14 / c)
    feats[:, 5]  = bb_pctb
    feats[:, 6]  = bb_bwp
    feats[:, 7]  = np.where(ema9  == 0, np.nan, (c - ema9)  / ema9  * 100.0)
    feats[:, 8]  = np.where(ema21 == 0, np.nan, (c - ema21) / ema21 * 100.0)
    feats[:, 9]  = np.where(ema50 == 0, np.nan, (c - ema50) / ema50 * 100.0)
    feats[:, 10] = ret(c, 1)
    feats[:, 11] = ret(c, 3)
    feats[:, 12] = ret(c, 5)
    feats[:, 13] = ret(c, 10)
    feats[:, 14] = ret(c, 20)
    feats[:, 15] = vol_rt
    feats[:, 16] = hl_r
    feats[:, 17] = cl_p
    feats[:, 18] = bd_r
    feats[:, 19] = will

    return feats, atr14


def agregar_symbol_id(X_base, symbol_name, n_symbols=4):
    n = X_base.shape[0]
    sym_id = SYMBOL_IDS.get(symbol_name, 0)
    one_hot = np.zeros((n, n_symbols), dtype=np.float64)
    one_hot[:, sym_id] = 1.0
    return np.hstack([X_base, one_hot])


# ╔══════════════════════════════════════════════════════════╗
# ║  MOTOR DE BACKTESTING                                    ║
# ╚══════════════════════════════════════════════════════════╝

class Trade:
    __slots__ = ['entry_price', 'sl', 'tp', 'direction', 'lots',
                 'entry_bar', 'exit_bar', 'exit_price', 'pnl',
                 'commission', 'status']

    def __init__(self, entry_price, sl, tp, direction, lots, entry_bar):
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.direction = direction  # 1 = buy, -1 = sell
        self.lots = lots
        self.entry_bar = entry_bar
        self.exit_bar = None
        self.exit_price = None
        self.pnl = 0.0
        self.commission = 0.0
        self.status = 'open'


def run_backtest(df, predictions, atr_values, simbolo, threshold_buy, threshold_sell, params=None):
    """
    Ejecuta backtest barra-a-barra simulando el EA real.

    Returns:
        trades: lista de Trade objects
        equity_curve: array con el equity en cada barra
    """
    if params is None:
        params = DEFAULT_PARAMS

    spread = SPREAD_PIPS.get(simbolo, 0.0001)
    slippage = SLIPPAGE_PIPS.get(simbolo, 0.00003)
    commission = COMMISSION_PER_LOT.get(simbolo, 7.0)

    sl_mult = params["sl_atr_mult"]
    tp_mult = params["tp_atr_mult"]
    risk_pct = params["risk_pct"]
    max_trades = params["max_trades"]

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(closes)

    balance = INITIAL_BALANCE
    equity_curve = np.full(n, INITIAL_BALANCE)
    open_trades = []
    closed_trades = []

    for i in range(60, n):
        atr_val = atr_values[i]
        if np.isnan(atr_val) or atr_val <= 0:
            equity_curve[i] = balance
            continue

        # ── Verificar SL/TP de trades abiertos ──
        for trade in open_trades[:]:
            hit_sl = False
            hit_tp = False

            if trade.direction == 1:  # BUY
                # Check SL
                if lows[i] <= trade.sl:
                    trade.exit_price = trade.sl
                    hit_sl = True
                # Check TP
                elif highs[i] >= trade.tp:
                    trade.exit_price = trade.tp
                    hit_tp = True
            else:  # SELL
                if highs[i] >= trade.sl:
                    trade.exit_price = trade.sl
                    hit_sl = True
                elif lows[i] <= trade.tp:
                    trade.exit_price = trade.tp
                    hit_tp = True

            if hit_sl or hit_tp:
                trade.exit_bar = i
                trade.status = 'closed'
                # PnL en unidades de precio
                price_diff = (trade.exit_price - trade.entry_price) * trade.direction
                # Convertir a USD (simplificado: asumimos lot_value standard)
                lot_value = _get_lot_value(simbolo, trade.entry_price)
                trade.pnl = price_diff * trade.lots * lot_value - trade.commission
                balance += trade.pnl
                closed_trades.append(trade)
                open_trades.remove(trade)

        # ── Generar señal si hay predicción válida ──
        pred = predictions[i]
        if np.isnan(pred):
            equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i], simbolo)
            continue

        n_open = len(open_trades)

        if pred > threshold_buy and n_open < max_trades:
            # Señal de compra
            entry = closes[i] + spread / 2 + slippage
            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult
            sl = entry - sl_dist
            tp = entry + tp_dist
            lots = _calc_lots(balance, risk_pct, sl_dist, simbolo, entry)
            if lots > 0:
                t = Trade(entry, sl, tp, 1, lots, i)
                t.commission = commission * lots
                open_trades.append(t)

        elif pred < threshold_sell and n_open < max_trades:
            # Señal de venta
            entry = closes[i] - spread / 2 - slippage
            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult
            sl = entry + sl_dist
            tp = entry - tp_dist
            lots = _calc_lots(balance, risk_pct, sl_dist, simbolo, entry)
            if lots > 0:
                t = Trade(entry, sl, tp, -1, lots, i)
                t.commission = commission * lots
                open_trades.append(t)

        equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i], simbolo)

    # Cerrar trades abiertos al final del período
    for trade in open_trades:
        trade.exit_price = closes[-1]
        trade.exit_bar = n - 1
        trade.status = 'forced_close'
        price_diff = (trade.exit_price - trade.entry_price) * trade.direction
        lot_value = _get_lot_value(simbolo, trade.entry_price)
        trade.pnl = price_diff * trade.lots * lot_value - trade.commission
        balance += trade.pnl
        closed_trades.append(trade)

    equity_curve[-1] = balance
    return closed_trades, equity_curve


def _get_lot_value(simbolo, price):
    """Retorna el valor de 1 lote en USD por unidad de movimiento de precio."""
    if simbolo == "xauusd":
        return 100        # 1 lote XAUUSD = 100 oz, $1 movimiento = $100
    elif simbolo == "usdjpy":
        return 100000 / price  # Convertir a USD
    else:
        return 100000     # Standard forex lot


def _calc_lots(balance, risk_pct, sl_distance, simbolo, price):
    """Calcula tamaño de lote basado en riesgo fijo."""
    risk_usd = balance * risk_pct / 100.0
    lot_value = _get_lot_value(simbolo, price)

    if sl_distance <= 0 or lot_value <= 0:
        return 0.0

    risk_per_lot = sl_distance * lot_value
    if risk_per_lot <= 0:
        return 0.0

    lots = risk_usd / risk_per_lot

    # Respetar límites
    min_lot = 0.01
    max_lot = 10.0
    step = 0.01
    lots = max(min_lot, min(max_lot, lots))
    lots = round(lots / step) * step
    return lots


def _unrealized_pnl(open_trades, current_price, simbolo):
    """Calcula PnL no realizado de posiciones abiertas."""
    total = 0.0
    for t in open_trades:
        lot_value = _get_lot_value(simbolo, t.entry_price)
        diff = (current_price - t.entry_price) * t.direction
        total += diff * t.lots * lot_value - t.commission
    return total


# ╔══════════════════════════════════════════════════════════╗
# ║  CÁLCULO DE MÉTRICAS FINANCIERAS                         ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_metricas(trades, equity_curve, initial_balance=INITIAL_BALANCE):
    """Calcula métricas financieras completas."""
    if not trades:
        return {
            "total_trades": 0, "net_profit": 0, "return_pct": 0,
            "win_rate": 0, "profit_factor": 0, "sharpe_ratio": 0,
            "sortino_ratio": 0, "max_drawdown_pct": 0, "max_drawdown_duration": 0,
            "calmar_ratio": 0, "avg_win": 0, "avg_loss": 0,
            "expectancy": 0, "trades_per_month": 0,
        }

    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    # Básicas
    net_profit = np.sum(pnls)
    return_pct = (equity_curve[-1] - initial_balance) / initial_balance * 100

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancia por trade
    expectancy = np.mean(pnls) if n_trades > 0 else 0

    # Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_dd_pct = abs(np.min(drawdown))

    # Duración del max drawdown (en barras)
    in_dd = equity_curve < running_max
    dd_duration = 0
    max_dd_duration = 0
    for is_dd in in_dd:
        if is_dd:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Sharpe Ratio (anualizado, asumiendo M15 = 4 barras/hora * ~6000 hrs/año)
    # Aproximación: 252 días * 24 horas * 4 = 24,192 barras M15/año
    equity_returns = np.diff(equity_curve) / equity_curve[:-1]
    equity_returns = equity_returns[equity_returns != 0]  # Filtrar barras sin cambio
    if len(equity_returns) > 1 and np.std(equity_returns) > 0:
        sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(24192)
    else:
        sharpe = 0.0

    # Sortino Ratio (solo penaliza downside)
    downside = equity_returns[equity_returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = np.mean(equity_returns) / np.std(downside) * np.sqrt(24192)
    else:
        sortino = 0.0

    # Calmar Ratio
    years = len(equity_curve) / 24192
    annual_return = return_pct / years if years > 0 else 0
    calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0

    # Trades por mes
    bars_per_month = 24192 / 12
    months = len(equity_curve) / bars_per_month
    trades_per_month = n_trades / months if months > 0 else 0

    return {
        "total_trades": n_trades,
        "net_profit": round(net_profit, 2),
        "return_pct": round(return_pct, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_bars": max_dd_duration,
        "calmar_ratio": round(calmar, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "trades_per_month": round(trades_per_month, 1),
        "annual_return_pct": round(annual_return, 2),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  BACKTESTING HybridAI v2.0 - Motor de Simulacion")
    print("=" * 65)

    # Cargar modelo ONNX
    ruta_onnx = f"{CARPETA_MODELO}/hybrid_ai_model.onnx"
    ruta_cfg  = f"{CARPETA_MODELO}/modelo_config.json"

    if not os.path.exists(ruta_onnx):
        print(f"\n  ERROR: No se encontro el modelo ONNX")
        print(f"  -> Ejecuta primero: python 3_entrenar_modelo.py")
        exit(1)

    sess = rt.InferenceSession(ruta_onnx)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Cargar config con umbrales óptimos
    config = {}
    if os.path.exists(ruta_cfg):
        with open(ruta_cfg) as f:
            config = json.load(f)

    umbrales = config.get("umbrales_por_simbolo", {})

    print(f"\n  Modelo cargado: {ruta_onnx}")
    print(f"  Balance inicial: ${INITIAL_BALANCE:,.0f}")

    all_results = {}
    combined_equity = None

    for simbolo in SIMBOLOS:
        archivo = f"{CARPETA_DATOS}/{simbolo}_m15.csv"
        if not os.path.exists(archivo):
            print(f"\n  {archivo} no encontrado, saltando...")
            continue

        print(f"\n{'='*65}")
        print(f"  BACKTESTING: {simbolo.upper()}")
        print(f"{'='*65}")

        df = pd.read_csv(archivo, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[df["high"] >= df["low"]]

        c = df["close"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        v = df["volume"].values.astype(np.float64)

        # Calcular features
        X_base, atr_values = calcular_features_array(c, h, l, o, v)
        X_full = agregar_symbol_id(X_base, simbolo)

        # Solo usar 20% final para backtest (out-of-sample)
        test_start = int(len(df) * 0.80)
        df_test = df.iloc[test_start:]
        X_test = X_full[test_start:]
        atr_test = atr_values[test_start:]

        print(f"  Periodo test: {df_test.index[0].date()} -> {df_test.index[-1].date()}")
        print(f"  Barras: {len(df_test):,}")

        # Generar predicciones
        predictions = np.full(len(df_test), np.nan)
        batch_size = 1000
        for start in range(0, len(X_test), batch_size):
            end = min(start + batch_size, len(X_test))
            batch = X_test[start:end].astype(np.float32)
            # Filtrar filas con NaN
            valid_mask = ~np.any(np.isnan(batch), axis=1)
            if valid_mask.any():
                valid_batch = batch[valid_mask]
                preds = sess.run([output_name], {input_name: valid_batch})[0].flatten()
                predictions[start:end][valid_mask] = preds

        # Obtener umbrales
        sym_upper = simbolo.upper()
        thr = umbrales.get(sym_upper, 0.15)
        threshold_buy = thr
        threshold_sell = -thr

        print(f"  Umbral: +/-{thr}%")
        print(f"  Predicciones validas: {(~np.isnan(predictions)).sum():,}")

        # Ejecutar backtest
        trades, equity = run_backtest(
            df_test, predictions, atr_test, simbolo,
            threshold_buy, threshold_sell
        )

        # Calcular métricas
        metrics = calcular_metricas(trades, equity)
        all_results[sym_upper] = metrics

        # Imprimir resultados
        print(f"\n  --- Resultados {sym_upper} ---")
        print(f"  Total trades:      {metrics['total_trades']}")
        print(f"  Profit neto:       ${metrics['net_profit']:+,.2f}")
        print(f"  Retorno:           {metrics['return_pct']:+.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
        print(f"  Avg Win:           ${metrics['avg_win']:+.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:+.2f}")
        print(f"  Expectancia:       ${metrics['expectancy']:+.2f}/trade")
        print(f"  Trades/mes:        {metrics['trades_per_month']:.1f}")

        if combined_equity is None:
            combined_equity = equity.copy()
        else:
            min_len = min(len(combined_equity), len(equity))
            combined_equity = combined_equity[:min_len]
            eq_contrib = equity[:min_len] - INITIAL_BALANCE
            combined_equity = combined_equity + eq_contrib

    # ── RESUMEN COMPARATIVO ──
    print(f"\n\n{'='*65}")
    print(f"  RESUMEN COMPARATIVO - TODOS LOS SIMBOLOS")
    print(f"{'='*65}")
    print(f"  {'Simbolo':<10} {'Trades':>7} {'Ret%':>8} {'WinR%':>7} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>8} {'$/trade':>9}")
    print(f"  {'-'*63}")

    for sym, m in all_results.items():
        status = "+" if m['net_profit'] > 0 else "-"
        print(f"  {sym:<10} {m['total_trades']:>7} {m['return_pct']:>+8.1f} "
              f"{m['win_rate']:>7.1f} {m['profit_factor']:>6.2f} "
              f"{m['sharpe_ratio']:>7.2f} {m['max_drawdown_pct']:>8.2f} "
              f"{m['expectancy']:>+9.2f}")

    # Guardar resultados
    results_path = f"{CARPETA_MODELO}/backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {results_path}")
    print(f"\n  Proximo paso:")
    print(f"    python 5_walk_forward.py  (validacion walk-forward)")
    print("=" * 65)
