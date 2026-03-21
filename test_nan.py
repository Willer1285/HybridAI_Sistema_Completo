import pandas as pd
import numpy as np

df = pd.read_csv("datos/xauusd_m15.csv")

c  = df["close"].values.astype(np.float64)
h  = df["high"].values.astype(np.float64)
l  = df["low"].values.astype(np.float64)
o  = df["open"].values.astype(np.float64)
v  = df["volume"].values.astype(np.float64)
n  = len(c)

def atr_calc(h_, l_, c_, period):
    tr = np.maximum(h_[1:] - l_[1:],
         np.maximum(np.abs(h_[1:] - c_[:-1]),
                    np.abs(l_[1:] - c_[:-1])))
    tr = np.concatenate([[np.nan], tr])
    result = np.full(n, np.nan)
    result[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result

atr = atr_calc(h, l, c, 14)
print("ATR is nan?", np.all(np.isnan(atr[20:])))

# Check all logic manually
import sys
sys.path.append(".")
import importlib

# To avoid full execution, copy the calculate function directly
def ema_calc(data, period):
    result = np.full(len(data), np.nan)
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1.0 - k)
    return result

ema9 = ema_calc(c, 9)
print("EMA9 has nan at end?", np.isnan(ema9[-1]))

# Load full module
import importlib.util
spec = importlib.util.spec_from_file_location("mod", "3_entrenar_modelo.py")
mod = importlib.util.module_from_spec(spec)
# Patching exit and print to avoid running
def noprint(*args, **kwargs): pass
import builtins
orig_print = builtins.print
builtins.print = noprint

try:
    spec.loader.exec_module(mod)
except Exception:
    pass

builtins.print = orig_print

feats = mod.calcular_features(df)
print("Features shape:", feats.shape)

for col in range(20):
    valido_col = ~np.isnan(feats[:, col])
    print(f"Col {col} valid count: {np.sum(valido_col)} / {len(feats)}")

