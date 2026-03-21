import MetaTrader5 as mt5

mt5.initialize()

print("MaxBars allowed in terminal:", mt5.terminal_info().maxbars)

for qty in [130000, 120000, 100000, 50000]:
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, qty)
    if rates is None:
        print(f"Failed to get {qty} bars. Error:", mt5.last_error())
    else:
        print(f"Successfully got {len(rates)} bars when requesting {qty}.")
        break

mt5.shutdown()
