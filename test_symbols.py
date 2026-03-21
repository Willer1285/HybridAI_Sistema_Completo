import MetaTrader5 as mt5

if not mt5.initialize():
    print("Failed to initialize MT5")
    exit(1)

# print all symbols containing USD
symbols = mt5.symbols_get("*USD*")
if symbols:
    print("USD Symbols:", [s.name for s in symbols][:50])
else:
    # If wildcard doesn't work, just get all and filter
    all_sym = mt5.symbols_get()
    usd_sym = [s.name for s in all_sym if 'USD' in s.name][:50]
    print("USD Symbols:", usd_sym)

# try to get 1 bar of EURUSD
sym = mt5.symbols_get("*EURUSD*")
if sym:
    name = sym[0].name
    mt5.symbol_select(name, True)
    rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M15, 0, 10)
    print(f"Rates for {name}:", rates)
else:
    print("No EURUSD found")

mt5.shutdown()
