import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # Use the classic version you installed
from binance.client import Client
import config

# Initialize Client
client = Client()

def fetchData(symbol="BTC", amount=1, timeframe="1d", as_csv=False, file_name=None):
    """
    Fetches OHLCV data from Binance and adds simplified technical indicators.
    Returns: Pandas DataFrame with Date, OHLCV, Returns, and 4 Key Indicators.
    """
    
    # 1. Simplified Timeframe Logic
    tf_ms = {
        '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000,
        '1h': 3600000, '2h': 7200000, '4h': 14400000, '6h': 21600000, '8h': 28800000,
        '12h': 43200000, '1d': 86400000, '3d': 259200000, '1w': 604800000, '1M': 2629800000
    }
    
    if timeframe not in tf_ms:
        print(f"Error: {timeframe} is an invalid timeframe.")
        return None

    full_symbol = symbol + "USDT"
    

    print("fetch data executed")

    # 2. Fetch Data
    candles = client.get_klines(symbol=full_symbol, limit=1000, interval=timeframe)
    
    if amount > 1:
        end_ts = candles[0][0]
        for _ in range(amount):
            prev_batch = client.get_klines(symbol=full_symbol, limit=1000, interval=timeframe, endTime=end_ts)
            candles = prev_batch + candles
            end_ts = prev_batch[0][0]

    # 3. Create DataFrame
    df = pd.DataFrame(candles)
    df = df.iloc[:, :6] 
    df.columns = ["date", "open", "high", "low", "close", "volume"]

    df["date"] = pd.to_datetime(df["date"], unit="ms")
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].apply(pd.to_numeric)

    # 4. Calculate Custom Metrics
    df["log_ret_close"] = np.log(df["close"]).diff()
    df["log_ret_vol"] = np.log(df["volume"].replace(0, np.nan)).diff()
    # Today's volume divided by the average of the PREVIOUS 7 days
    df["volatility"] = df["volume"] / df["volume"].shift(1).rolling(window=7).mean()

    # 5. Technical Indicators (Simplified for Model Input)
    # We purposefully do NOT use append=True here so we can grab specific columns.

    # A. RSI (14) -> Standard 0-100 scale
    df['rsi'] = df.ta.rsi(length=14)

    # B. MACD (12, 26, 9) -> We extract ONLY the Histogram (MACDh)
    # This represents momentum (positive = bullish, negative = bearish)
    macd_full = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd_full['MACDh_12_26_9']

    # C. Bollinger Bands (20, 2) -> We extract ONLY Percent B (BBP)
    # 0.0 = Price at Lower Band, 0.5 = Middle, 1.0 = Upper Band
    bb_full = df.ta.bbands(length=20, std=2)
    df['bollinger_bands'] = bb_full['BBP_20_2.0']

    # D. ATR (14) -> Standard Volatility
    df['atr'] = df.ta.atr(length=14)

    # 6. Final Cleanup
    # This drops the first ~33 rows needed for MACD/Bollinger warm-up
    df.dropna(inplace=True)

    if as_csv:
        if file_name is None:
            file_name = f"{full_symbol}_{timeframe}.csv"
        try:
            df.to_csv(f"{config.data_path}/{symbol}/{file_name}", index=False)
            print(f"Successfully saved {len(df)} rows to {file_name}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    return df