# =============================================================
#  PASO 4 - BACKTESTING v3.0 CON PROTECCIONES ANTI-CRASH
#  Solo XAUUSD. Incluye:
#   - Drawdown circuit breaker (detiene trading si DD > 20%)
#   - Posición máxima (cap de lotes)
#   - Filtro de régimen de volatilidad
#   - Cooling period tras pérdidas consecutivas
#
#  Comando: python 4_backtesting.py
# =============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

import onnxruntime as rt

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"
N_FEATURES_BASE = 20
N_FEATURES      = 24
BARRAS_FUTURO   = 5

# Costos XAUUSD
SPREAD          = 0.30     # $0.30 en gold
COMMISSION_LOT  = 7.0      # $7 por lote round-trip
SLIPPAGE        = 0.10     # $0.10

# Trading defaults
DEFAULT_PARAMS = {
    "sl_atr_mult":  2.0,
    "tp_atr_mult":  3.5,
    "risk_pct":     1.0,
    "max_trades":   1,
}

# ── PROTECCIONES v3.0 ──
MAX_DRAWDOWN_PCT     = 20.0    # Circuit breaker: pausar si DD > 20%
DD_RESUME_PCT        = 10.0    # Reanudar si DD recupera a < 10%
MAX_LOTS             = 5.0     # Nunca más de 5 lotes por trade
MAX_CONSEC_LOSSES    = 3       # Cooling tras N pérdidas seguidas
COOLING_BARS         = 5       # Barras de espera tras cooling
VOL_REGIME_MIN       = 0.6    # No operar si volatilidad muy baja
VOL_REGIME_MAX       = 1.8    # No operar si volatilidad muy alta

INITIAL_BALANCE = 10000.0
LOT_VALUE_GOLD  = 100          # 1 lote XAUUSD = 100 oz
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE FEATURES v3.0                              ║
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


def calcular_features_v3(df):
    """Calcula los 24 features v3.0 para XAUUSD."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)

    feats = np.full((n, N_FEATURES), np.nan, dtype=np.float64)

    # RSI(14)
    rsi_p = 14
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n, np.nan); al = np.full(n, np.nan)
    ag[rsi_p] = np.mean(gain[:rsi_p]); al[rsi_p] = np.mean(loss[:rsi_p])
    for i in range(rsi_p + 1, n):
        ag[i] = (ag[i-1] * (rsi_p-1) + gain[i-1]) / rsi_p
        al[i] = (al[i-1] * (rsi_p-1) + loss[i-1]) / rsi_p
    rs = np.where(al == 0, 100.0, ag / al)
    rsi = 100.0 - (100.0 / (1.0 + rs)); rsi[:rsi_p] = np.nan

    # MACD
    ema12 = ema_calc(c, 12); ema26 = ema_calc(c, 26)
    macd_line = ema12 - ema26
    macd_sig = ema_calc(macd_line, 9); macd_hist = macd_line - macd_sig

    # ATR
    atr14 = atr_calc(h, l, c, 14)
    atr_s = np.where(atr14 == 0, 1e-10, atr14)

    # Bollinger
    sma20 = sma_calc(c, 20)
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(c[i-19:i+1], ddof=0)
    bb_up = sma20 + 2.0 * std20; bb_lo = sma20 - 2.0 * std20
    bb_w = bb_up - bb_lo
    bb_pctb = np.where(bb_w == 0, 0.5, (c - bb_lo) / bb_w)
    bb_bwp = np.where(sma20 == 0, np.nan, bb_w / sma20)

    # EMAs
    ema9 = ema_calc(c, 9); ema21 = ema_calc(c, 21); ema50 = ema_calc(c, 50)

    def ret(data, p):
        r = np.full(n, np.nan)
        r[p:] = (data[p:] - data[:-p]) / data[:-p] * 100.0
        return r

    vol_ma = sma_calc(v, 20)
    vol_rt = np.where(vol_ma == 0, 1.0, v / vol_ma)

    hl_r = np.where(c == 0, np.nan, (h - l) / c * 100.0)
    rng = h - l
    cl_p = np.where(rng == 0, 0.5, (c - l) / rng)
    bd_r = np.where(rng == 0, 0.0, (c - o) / rng)

    mxh = np.full(n, np.nan); mnl = np.full(n, np.nan)
    for i in range(13, n):
        mxh[i] = np.max(h[i-13:i+1]); mnl[i] = np.min(l[i-13:i+1])
    wl_r = mxh - mnl
    will = np.where(wl_r == 0, 0.5, (c - mnl) / wl_r)

    # Base 20
    feats[:, 0] = rsi/100.0; feats[:, 1] = macd_line/atr_s; feats[:, 2] = macd_sig/atr_s
    feats[:, 3] = macd_hist/atr_s; feats[:, 4] = np.where(c==0, np.nan, atr14/c)
    feats[:, 5] = bb_pctb; feats[:, 6] = bb_bwp
    feats[:, 7] = np.where(ema9==0, np.nan, (c-ema9)/ema9*100.0)
    feats[:, 8] = np.where(ema21==0, np.nan, (c-ema21)/ema21*100.0)
    feats[:, 9] = np.where(ema50==0, np.nan, (c-ema50)/ema50*100.0)
    feats[:, 10] = ret(c,1); feats[:, 11] = ret(c,3); feats[:, 12] = ret(c,5)
    feats[:, 13] = ret(c,10); feats[:, 14] = ret(c,20)
    feats[:, 15] = vol_rt; feats[:, 16] = hl_r; feats[:, 17] = cl_p
    feats[:, 18] = bd_r; feats[:, 19] = will

    # v3.0: 4 nuevos
    atr_sma50 = sma_calc(atr14, 50)
    vol_regime = np.where((atr_sma50 == 0) | np.isnan(atr_sma50), 1.0, atr14 / atr_sma50)
    feats[:, 20] = vol_regime
    feats[:, 21] = np.where(atr_s == 0, 0.0, (ema21 - ema50) / atr_s)

    hours = df.index.hour + df.index.minute / 60.0
    feats[:, 22] = np.sin(2.0 * np.pi * hours / 24.0)
    feats[:, 23] = np.cos(2.0 * np.pi * hours / 24.0)

    return feats, atr14, vol_regime


# ╔══════════════════════════════════════════════════════════╗
# ║  MOTOR DE BACKTESTING v3.0 CON PROTECCIONES              ║
# ╚══════════════════════════════════════════════════════════╝

class Trade:
    __slots__ = ['entry_price', 'sl', 'tp', 'direction', 'lots',
                 'entry_bar', 'exit_bar', 'exit_price', 'pnl',
                 'commission', 'status']

    def __init__(self, entry_price, sl, tp, direction, lots, entry_bar):
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.direction = direction
        self.lots = lots
        self.entry_bar = entry_bar
        self.exit_bar = None
        self.exit_price = None
        self.pnl = 0.0
        self.commission = 0.0
        self.status = 'open'


def run_backtest(df, predictions, atr_values, vol_regime, threshold, params=None):
    """Backtest v3.0 con protecciones anti-crash."""
    if params is None:
        params = DEFAULT_PARAMS

    sl_mult = params["sl_atr_mult"]
    tp_mult = params["tp_atr_mult"]
    risk_pct = params["risk_pct"]
    max_trades = params["max_trades"]

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(closes)

    balance = INITIAL_BALANCE
    peak_equity = INITIAL_BALANCE
    equity_curve = np.full(n, INITIAL_BALANCE)
    open_trades = []
    closed_trades = []

    # Estado de protecciones
    is_suspended = False       # Circuit breaker activo
    consecutive_losses = 0     # Contador de pérdidas seguidas
    cooling_until = 0          # Barra hasta la que hay cooling
    suspension_events = 0      # Cuántas veces se activó el circuit breaker

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
                if lows[i] <= trade.sl:
                    trade.exit_price = trade.sl
                    hit_sl = True
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
                price_diff = (trade.exit_price - trade.entry_price) * trade.direction
                trade.pnl = price_diff * trade.lots * LOT_VALUE_GOLD - trade.commission
                balance += trade.pnl
                closed_trades.append(trade)
                open_trades.remove(trade)

                # Tracking de pérdidas consecutivas
                if trade.pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                # Cooling after consecutive losses
                if consecutive_losses >= MAX_CONSEC_LOSSES:
                    cooling_until = i + COOLING_BARS
                    consecutive_losses = 0

        # ── Actualizar peak equity ──
        current_equity = balance + _unrealized_pnl(open_trades, closes[i])
        if current_equity > peak_equity:
            peak_equity = current_equity

        # ── CIRCUIT BREAKER: Drawdown check ──
        if peak_equity > 0:
            dd_pct = (peak_equity - current_equity) / peak_equity * 100.0
            if dd_pct >= MAX_DRAWDOWN_PCT and not is_suspended:
                is_suspended = True
                suspension_events += 1
            elif is_suspended and dd_pct < DD_RESUME_PCT:
                is_suspended = False

        # ── No abrir nuevos trades si protecciones activas ──
        if is_suspended or i < cooling_until:
            equity_curve[i] = current_equity
            continue

        # ── Filtro de régimen de volatilidad ──
        vr = vol_regime[i] if i < len(vol_regime) else 1.0
        if np.isnan(vr) or vr < VOL_REGIME_MIN or vr > VOL_REGIME_MAX:
            equity_curve[i] = current_equity
            continue

        # ── Generar señal ──
        pred = predictions[i]
        if np.isnan(pred):
            equity_curve[i] = current_equity
            continue

        n_open = len(open_trades)

        if pred > threshold and n_open < max_trades:
            entry = closes[i] + SPREAD / 2 + SLIPPAGE
            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult
            sl = entry - sl_dist
            tp = entry + tp_dist
            lots = _calc_lots(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, sl, tp, 1, lots, i)
                t.commission = COMMISSION_LOT * lots
                open_trades.append(t)

        elif pred < -threshold and n_open < max_trades:
            entry = closes[i] - SPREAD / 2 - SLIPPAGE
            sl_dist = atr_val * sl_mult
            tp_dist = atr_val * tp_mult
            sl = entry + sl_dist
            tp = entry - tp_dist
            lots = _calc_lots(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, sl, tp, -1, lots, i)
                t.commission = COMMISSION_LOT * lots
                open_trades.append(t)

        equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i])

    # Cerrar trades abiertos al final
    for trade in open_trades:
        trade.exit_price = closes[-1]
        trade.exit_bar = n - 1
        trade.status = 'forced_close'
        price_diff = (trade.exit_price - trade.entry_price) * trade.direction
        trade.pnl = price_diff * trade.lots * LOT_VALUE_GOLD - trade.commission
        balance += trade.pnl
        closed_trades.append(trade)

    equity_curve[-1] = balance
    return closed_trades, equity_curve, suspension_events


def _calc_lots(balance, risk_pct, sl_distance):
    """Calcula lotes con cap máximo."""
    risk_usd = balance * risk_pct / 100.0
    risk_per_lot = sl_distance * LOT_VALUE_GOLD
    if risk_per_lot <= 0:
        return 0.0
    lots = risk_usd / risk_per_lot
    lots = max(0.01, min(MAX_LOTS, lots))  # CAP en MAX_LOTS
    lots = round(lots / 0.01) * 0.01
    return lots


def _unrealized_pnl(open_trades, current_price):
    total = 0.0
    for t in open_trades:
        diff = (current_price - t.entry_price) * t.direction
        total += diff * t.lots * LOT_VALUE_GOLD - t.commission
    return total


# ╔══════════════════════════════════════════════════════════╗
# ║  MÉTRICAS FINANCIERAS                                    ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_metricas(trades, equity_curve, initial_balance=INITIAL_BALANCE):
    if not trades:
        return {k: 0 for k in [
            "total_trades", "net_profit", "return_pct", "win_rate",
            "profit_factor", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "calmar_ratio", "avg_win", "avg_loss",
            "expectancy", "trades_per_month",
        ]}

    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)
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
    expectancy = np.mean(pnls) if n_trades > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_dd_pct = abs(np.min(drawdown))

    equity_returns = np.diff(equity_curve) / equity_curve[:-1]
    equity_returns = equity_returns[equity_returns != 0]
    if len(equity_returns) > 1 and np.std(equity_returns) > 0:
        sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(24192)
    else:
        sharpe = 0.0

    downside = equity_returns[equity_returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = np.mean(equity_returns) / np.std(downside) * np.sqrt(24192)
    else:
        sortino = 0.0

    years = len(equity_curve) / 24192
    annual_return = return_pct / years if years > 0 else 0
    calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0

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
        "calmar_ratio": round(calmar, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "trades_per_month": round(trades_per_month, 1),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  BACKTEST SIN PROTECCIONES (comparación)                  ║
# ╚══════════════════════════════════════════════════════════╝

def _calc_lots_no_cap(balance, risk_pct, sl_distance):
    """Lotes SIN cap (para comparación)."""
    risk_usd = balance * risk_pct / 100.0
    risk_per_lot = sl_distance * LOT_VALUE_GOLD
    if risk_per_lot <= 0:
        return 0.0
    lots = risk_usd / risk_per_lot
    lots = max(0.01, min(100.0, lots))
    lots = round(lots / 0.01) * 0.01
    return lots


def run_backtest_no_protection(df, predictions, atr_values, threshold, params=None):
    """Backtest SIN protecciones para comparación."""
    if params is None:
        params = DEFAULT_PARAMS

    sl_mult = params["sl_atr_mult"]
    tp_mult = params["tp_atr_mult"]
    risk_pct = params["risk_pct"]
    max_trades = params["max_trades"]

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
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

        for trade in open_trades[:]:
            hit_sl = hit_tp = False
            if trade.direction == 1:
                if lows[i] <= trade.sl:
                    trade.exit_price = trade.sl; hit_sl = True
                elif highs[i] >= trade.tp:
                    trade.exit_price = trade.tp; hit_tp = True
            else:
                if highs[i] >= trade.sl:
                    trade.exit_price = trade.sl; hit_sl = True
                elif lows[i] <= trade.tp:
                    trade.exit_price = trade.tp; hit_tp = True

            if hit_sl or hit_tp:
                trade.exit_bar = i; trade.status = 'closed'
                price_diff = (trade.exit_price - trade.entry_price) * trade.direction
                trade.pnl = price_diff * trade.lots * LOT_VALUE_GOLD - trade.commission
                balance += trade.pnl
                closed_trades.append(trade); open_trades.remove(trade)

        pred = predictions[i]
        if np.isnan(pred):
            equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i])
            continue

        n_open = len(open_trades)

        if pred > threshold and n_open < max_trades:
            entry = closes[i] + SPREAD / 2 + SLIPPAGE
            sl_dist = atr_val * sl_mult; tp_dist = atr_val * tp_mult
            lots = _calc_lots_no_cap(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, entry - sl_dist, entry + tp_dist, 1, lots, i)
                t.commission = COMMISSION_LOT * lots; open_trades.append(t)

        elif pred < -threshold and n_open < max_trades:
            entry = closes[i] - SPREAD / 2 - SLIPPAGE
            sl_dist = atr_val * sl_mult; tp_dist = atr_val * tp_mult
            lots = _calc_lots_no_cap(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, entry + sl_dist, entry - tp_dist, -1, lots, i)
                t.commission = COMMISSION_LOT * lots; open_trades.append(t)

        equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i])

    for trade in open_trades:
        trade.exit_price = closes[-1]; trade.exit_bar = n - 1
        trade.status = 'forced_close'
        price_diff = (trade.exit_price - trade.entry_price) * trade.direction
        trade.pnl = price_diff * trade.lots * LOT_VALUE_GOLD - trade.commission
        balance += trade.pnl; closed_trades.append(trade)

    equity_curve[-1] = balance
    return closed_trades, equity_curve, 0


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  BACKTESTING v3.0 - XAUUSD CON PROTECCIONES")
    print("=" * 65)

    ruta_onnx = f"{CARPETA_MODELO}/hybrid_ai_model.onnx"
    ruta_cfg = f"{CARPETA_MODELO}/modelo_config.json"

    if not os.path.exists(ruta_onnx):
        print(f"\n  ERROR: Modelo no encontrado -> ejecuta python 3_entrenar_modelo.py")
        exit(1)

    sess = rt.InferenceSession(ruta_onnx)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Cargar umbral
    threshold = 0.10
    if os.path.exists(ruta_cfg):
        with open(ruta_cfg) as f:
            cfg = json.load(f)
        threshold = cfg.get("umbral_optimo", 0.10)

    print(f"\n  Modelo: {ruta_onnx}")
    print(f"  Balance: ${INITIAL_BALANCE:,.0f}")
    print(f"  Umbral: +/-{threshold}%")
    print(f"\n  PROTECCIONES ACTIVAS:")
    print(f"    Circuit breaker: DD > {MAX_DRAWDOWN_PCT}% -> pausa")
    print(f"    Max lotes:       {MAX_LOTS}")
    print(f"    Vol. regime:     [{VOL_REGIME_MIN}, {VOL_REGIME_MAX}]")
    print(f"    Cooling:         {COOLING_BARS} barras tras {MAX_CONSEC_LOSSES} losses")

    archivo = f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv"
    if not os.path.exists(archivo):
        print(f"\n  ERROR: {archivo} no encontrado")
        exit(1)

    df = pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[df["high"] >= df["low"]]

    # Calcular features v3.0
    X_full, atr_values, vol_regime = calcular_features_v3(df)

    # Test en el 20% final (out-of-sample)
    test_start = int(len(df) * 0.80)
    df_test = df.iloc[test_start:]
    X_test = X_full[test_start:]
    atr_test = atr_values[test_start:]
    vol_regime_test = vol_regime[test_start:]

    print(f"\n{'='*65}")
    print(f"  BACKTESTING: XAUUSD")
    print(f"{'='*65}")
    print(f"  Periodo: {df_test.index[0].date()} -> {df_test.index[-1].date()}")
    print(f"  Barras: {len(df_test):,}")

    # Predicciones
    predictions = np.full(len(df_test), np.nan)
    batch_size = 1000
    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        batch = X_test[start:end].astype(np.float32)
        valid_mask = ~np.any(np.isnan(batch), axis=1)
        if valid_mask.any():
            valid_batch = batch[valid_mask]
            preds = sess.run([output_name], {input_name: valid_batch})[0].flatten()
            predictions[start:end][valid_mask] = preds

    print(f"  Predicciones validas: {(~np.isnan(predictions)).sum():,}")

    # Backtest CON protecciones
    trades, equity, n_suspensions = run_backtest(
        df_test, predictions, atr_test, vol_regime_test, threshold
    )

    metrics = calcular_metricas(trades, equity)

    print(f"\n  --- Resultados XAUUSD (CON protecciones) ---")
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
    print(f"  Circuit breakers:  {n_suspensions} activaciones")
    print(f"  Equity final:      ${equity[-1]:,.2f}")
    print(f"  Peak equity:       ${np.max(equity):,.2f}")

    # ── Backtest SIN protecciones (comparación) ──
    print(f"\n{'='*65}")
    print(f"  COMPARACION: SIN protecciones (referencia)")
    print(f"{'='*65}")

    # Guardar y temporalmente desactivar protecciones
    old_max_dd = MAX_DRAWDOWN_PCT
    old_max_lots = MAX_LOTS
    old_vol_min = VOL_REGIME_MIN
    old_vol_max = VOL_REGIME_MAX

    # Simular sin protecciones usando un backtest simple
    trades_np, equity_np, _ = run_backtest_no_protection(
        df_test, predictions, atr_test, threshold
    )
    metrics_np = calcular_metricas(trades_np, equity_np)

    print(f"  Trades:       {metrics_np['total_trades']}")
    print(f"  Profit:       ${metrics_np['net_profit']:+,.2f}")
    print(f"  Max DD:       {metrics_np['max_drawdown_pct']:.2f}%")
    print(f"  Peak equity:  ${np.max(equity_np):,.2f}")
    print(f"  Equity final: ${equity_np[-1]:,.2f}")

    if np.max(equity_np) > 0 and equity_np[-1] < np.max(equity_np) * 0.5:
        pct_lost = (np.max(equity_np) - equity_np[-1]) / np.max(equity_np) * 100
        print(f"\n  SIN protecciones perdio {pct_lost:.0f}% desde el pico!")
        print(f"  Las protecciones evitarian esta catastrofe en el EA.")

    # Guardar resultados
    all_results = {
        "XAUUSD_protegido": metrics,
        "XAUUSD_sin_proteccion": metrics_np,
        "protecciones": {
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "max_lots": MAX_LOTS,
            "vol_regime_range": [VOL_REGIME_MIN, VOL_REGIME_MAX],
            "cooling_bars": COOLING_BARS,
            "circuit_breaker_activations": n_suspensions,
        },
    }

    results_path = f"{CARPETA_MODELO}/backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {results_path}")
    print(f"\n  Proximo paso:")
    print(f"    python 5_walk_forward.py")
    print("=" * 65)
