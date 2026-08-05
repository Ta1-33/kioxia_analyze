import yfinance as yf
import pandas as pd
import numpy as np


TICKER = "285A.T"


def fetch_intraday() -> pd.DataFrame:
    """Fetch hourly data for the most recent trading day."""
    df = yf.download(TICKER, period="5d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if df.empty:
        return df
    # Filter to the most recent trading day (by date)
    last_date = df.index.normalize().max()
    df = df[df.index.normalize() == last_date]
    # Convert to Japan time if timezone-aware
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("Asia/Tokyo")
    return df


def fetch_data(period: str = "2y") -> pd.DataFrame:
    df = yf.download(TICKER, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    # Moving averages
    df["MA5"] = close.rolling(5).mean()
    df["MA25"] = close.rolling(25).mean()
    df["MA75"] = close.rolling(75).mean()

    # Bollinger Bands (25-day, ±2σ)
    bb_std = close.rolling(25).std()
    df["BB_upper"] = df["MA25"] + 2 * bb_std
    df["BB_lower"] = df["MA25"] - 2 * bb_std

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()

    # Daily return
    df["Return"] = close.pct_change()

    return df
