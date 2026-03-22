# =============================================================
#  PASO 4 - BACKTESTING v4.0 - CLASIFICACIÓN MULTI-TF
#  Usa el modelo de clasificación (BUY/SELL/NEUTRAL) con
#  features multi-timeframe.
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
import sys

# Importar funciones del entrenamiento
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

# ── CONFIGURACIÓN ────────────────────────────────────────────
SIMBOLO         = "xauusd"
CARPETA_DATOS   = "datos"
CARPETA_MODELO  = "modelo"

# Costos XAUUSD
SPREAD          = 0.30
COMMISSION_LOT  = 7.0
SLIPPAGE        = 0.10

DEFAULT_PARAMS = {
    "sl_atr_mult":  2.0,
    "tp_atr_mult":  3.5,
    "risk_pct":     1.0,
    "max_trades":   1,
}

INITIAL_BALANCE = 10000.0
LOT_VALUE_GOLD  = 100
MAX_LOTS        = 5.0
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════╗
# ║  MOTOR DE BACKTESTING v4.0                                ║
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


def _calc_lots(balance, risk_pct, sl_distance):
    risk_usd = balance * risk_pct / 100.0
    risk_per_lot = sl_distance * LOT_VALUE_GOLD
    if risk_per_lot <= 0:
        return 0.0
    lots = risk_usd / risk_per_lot
    lots = max(0.01, min(MAX_LOTS, lots))
    lots = round(lots / 0.01) * 0.01
    return lots


def _unrealized_pnl(open_trades, current_price):
    total = 0.0
    for t in open_trades:
        diff = (current_price - t.entry_price) * t.direction
        total += diff * t.lots * LOT_VALUE_GOLD - t.commission
    return total


def run_backtest(df, predictions, atr_values, params=None):
    """Backtest v4.0 para clasificación (0=neutral, 1=buy, 2=sell)."""
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

    for i in range(210, n):
        atr_val = atr_values[i]
        if np.isnan(atr_val) or atr_val <= 0:
            equity_curve[i] = balance
            continue

        # Verificar SL/TP de trades abiertos
        for trade in open_trades[:]:
            hit_sl = hit_tp = False
            if trade.direction == 1:
                if lows[i] <= trade.sl: trade.exit_price = trade.sl; hit_sl = True
                elif highs[i] >= trade.tp: trade.exit_price = trade.tp; hit_tp = True
            else:
                if highs[i] >= trade.sl: trade.exit_price = trade.sl; hit_sl = True
                elif lows[i] <= trade.tp: trade.exit_price = trade.tp; hit_tp = True

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

        clase = int(pred)
        n_open = len(open_trades)

        if clase == 1 and n_open < max_trades:  # BUY
            entry = closes[i] + SPREAD / 2 + SLIPPAGE
            sl_dist = atr_val * sl_mult; tp_dist = atr_val * tp_mult
            lots = _calc_lots(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, entry - sl_dist, entry + tp_dist, 1, lots, i)
                t.commission = COMMISSION_LOT * lots; open_trades.append(t)

        elif clase == 2 and n_open < max_trades:  # SELL
            entry = closes[i] - SPREAD / 2 - SLIPPAGE
            sl_dist = atr_val * sl_mult; tp_dist = atr_val * tp_mult
            lots = _calc_lots(balance, risk_pct, sl_dist)
            if lots > 0:
                t = Trade(entry, entry + sl_dist, entry - tp_dist, -1, lots, i)
                t.commission = COMMISSION_LOT * lots; open_trades.append(t)

        equity_curve[i] = balance + _unrealized_pnl(open_trades, closes[i])

    # Cerrar trades al final
    for trade in open_trades:
        trade.exit_price = closes[-1]; trade.exit_bar = n - 1
        trade.status = 'forced_close'
        price_diff = (trade.exit_price - trade.entry_price) * trade.direction
        trade.pnl = price_diff * trade.lots * LOT_VALUE_GOLD - trade.commission
        balance += trade.pnl; closed_trades.append(trade)

    equity_curve[-1] = balance
    return closed_trades, equity_curve


# ╔══════════════════════════════════════════════════════════╗
# ║  MÉTRICAS                                                ║
# ╚══════════════════════════════════════════════════════════╝

def calcular_metricas(trades, equity_curve, initial_balance=INITIAL_BALANCE):
    if not trades:
        return {k: 0 for k in [
            "total_trades", "net_profit", "return_pct", "win_rate",
            "profit_factor", "sharpe_ratio", "max_drawdown_pct",
            "avg_win", "avg_loss", "expectancy", "buy_trades", "sell_trades",
        ]}

    pnls = np.array([t.pnl for t in trades])
    dirs = np.array([t.direction for t in trades])
    n_trades = len(pnls)
    net_profit = np.sum(pnls)
    return_pct = (equity_curve[-1] - initial_balance) / initial_balance * 100

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n_trades * 100
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss
    expectancy = np.mean(pnls)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    max_dd_pct = abs(np.min(drawdown))

    equity_returns = np.diff(equity_curve) / equity_curve[:-1]
    equity_returns = equity_returns[equity_returns != 0]
    sharpe = 0.0
    if len(equity_returns) > 1 and np.std(equity_returns) > 0:
        sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(24192)

    return {
        "total_trades": n_trades,
        "buy_trades": int(np.sum(dirs == 1)),
        "sell_trades": int(np.sum(dirs == -1)),
        "net_profit": round(net_profit, 2),
        "return_pct": round(return_pct, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  EJECUCIÓN PRINCIPAL                                     ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 65)
    print("  BACKTESTING v4.0 - CLASIFICACIÓN MULTI-TF")
    print("=" * 65)

    # Importar funciones de features
    trainer = import_module("3_entrenar_modelo")

    ruta_onnx = f"{CARPETA_MODELO}/hybrid_ai_model.onnx"
    if not os.path.exists(ruta_onnx):
        print(f"\n  ERROR: Modelo no encontrado -> ejecuta python 3_entrenar_modelo.py")
        exit(1)

    sess = rt.InferenceSession(ruta_onnx)
    input_name = sess.get_inputs()[0].name

    # Cargar datos multi-TF
    archivos = {
        "m15": f"{CARPETA_DATOS}/{SIMBOLO}_m15.csv",
        "h1":  f"{CARPETA_DATOS}/{SIMBOLO}_h1.csv",
        "h4":  f"{CARPETA_DATOS}/{SIMBOLO}_h4.csv",
        "d1":  f"{CARPETA_DATOS}/{SIMBOLO}_d1.csv",
    }
    for path in archivos.values():
        if not os.path.exists(path):
            print(f"  ERROR: {path} no encontrado"); exit(1)

    dfs = {}
    for tf, path in archivos.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
        df = df[df["high"] >= df["low"]]
        dfs[tf] = df

    df_m15 = dfs["m15"]

    # Calcular features
    X_m15, atr_values = trainer.calcular_features_m15(df_m15)
    X_htf = trainer.calcular_features_htf(df_m15, dfs["h1"], dfs["h4"], dfs["d1"])
    X_full = np.hstack([X_m15, X_htf])

    # Test en el 20% final
    test_start = int(len(df_m15) * 0.80)
    df_test = df_m15.iloc[test_start:]
    X_test = X_full[test_start:]
    atr_test = atr_values[test_start:]

    print(f"\n  Periodo: {df_test.index[0].date()} -> {df_test.index[-1].date()}")
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
            preds = sess.run(None, {input_name: valid_batch})[0].flatten()
            predictions[start:end][valid_mask] = preds

    n_buy = np.sum(predictions == 1)
    n_sell = np.sum(predictions == 2)
    n_neutral = np.sum(predictions == 0)
    n_valid = n_buy + n_sell + n_neutral
    print(f"  Predicciones: BUY={n_buy:,}  SELL={n_sell:,}  NEUTRAL={n_neutral:,}")
    if n_buy + n_sell > 0:
        print(f"  Ratio BUY/SELL: {n_buy/(n_buy+n_sell)*100:.1f}% / {n_sell/(n_buy+n_sell)*100:.1f}%")

    # Backtest
    trades, equity = run_backtest(df_test, predictions, atr_test)
    metrics = calcular_metricas(trades, equity)

    print(f"\n{'='*65}")
    print(f"  RESULTADOS")
    print(f"{'='*65}")
    print(f"  Total trades:      {metrics['total_trades']}")
    print(f"    BUY:             {metrics['buy_trades']}")
    print(f"    SELL:            {metrics['sell_trades']}")
    print(f"  Profit neto:       ${metrics['net_profit']:+,.2f}")
    print(f"  Retorno:           {metrics['return_pct']:+.2f}%")
    print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Avg Win:           ${metrics['avg_win']:+.2f}")
    print(f"  Avg Loss:          ${metrics['avg_loss']:+.2f}")
    print(f"  Expectancia:       ${metrics['expectancy']:+.2f}/trade")
    print(f"  Equity final:      ${equity[-1]:,.2f}")

    # Guardar
    results_path = f"{CARPETA_MODELO}/backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados guardados: {results_path}")
    print(f"\n  Proximo paso:")
    print(f"    python 5_walk_forward.py")
    print("=" * 65)
