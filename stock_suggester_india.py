 
#!/usr/bin/env python3
"""
stock_suggester_india.py

India-first stock suggestion script:
- prefers jugaad-data / nsepython for bhavcopy/delivery% if available
- falls back to yfinance for price data
- uses India VIX (^INDIAVIX) as a contextual feature if available
- trains a LightGBM multiclass classifier with labels mapped to {0,1,2}:
    0 -> SELL   (future_return < -thr)
    1 -> HOLD   (|future_return| <= thr)
    2 -> BUY    (future_return > thr)
- outputs: STOCK: <TICKER> — SUGGESTION: BUY|HOLD|SELL — CONFIDENCE: xx.xx%
"""

import argparse
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import ta
import pytz

warnings.filterwarnings("ignore")

# Try India-specific libs (optional)
JUGAAD_OK = False
NSEPY_OK = False
try:
    from jugaad_data.nse import NseBhavCopy
    JUGAAD_OK = True
except Exception:
    JUGAAD_OK = False

try:
    import nsepython as nsp  # optional
    NSEPY_OK = True
except Exception:
    NSEPY_OK = False

ASIA_KOLKATA = pytz.timezone("Asia/Kolkata")


# -----------------------------
# Fetchers
# -----------------------------
def fetch_price_yahoo(ticker, period="3y", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_delivery_jugaad(symbol, start, end):
    """
    If jugaad-data available, attempt to fetch DELIV_PER series for symbol (without .NS suffix).
    Returns pandas Series indexed by date or None.
    """
    if not JUGAAD_OK:
        return None
    try:
        nb = NseBhavCopy()
        sym = symbol.replace(".NS", "").replace(".BO", "")
        df = nb.get(symbol=sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return None
        if "DELIV_PER" in df.columns:
            s = df["DELIV_PER"].copy()
            s.index = pd.to_datetime(s.index)
            s.name = "delivery_pct"
            return s
    except Exception:
        return None
    return None


def fetch_india_vix(period="3y"):
    """Attempt to fetch India VIX via Yahoo ticker '^INDIAVIX'."""
    try:
        t = yf.Ticker("^INDIAVIX")
        df = t.history(period=period, interval="1d")
        if df is None or df.empty:
            return None
        ser = df["Close"].copy()
        ser.index = pd.to_datetime(ser.index)
        ser.name = "india_vix"
        return ser
    except Exception:
        return None


def fetch_fii_dii_stub():
    """
    Placeholder for FII/DII ingestion. Return None by default.
    If you have a CSV or API for daily FII/DII flows, load it and return a DataFrame with
    columns ['date','fii_net','dii_net'] indexed by date.
    """
    return None


# -----------------------------
# Feature engineering
# -----------------------------
def add_technical_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    # Moving averages
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI, ATR, volatility
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["vol_20"] = df["return"].rolling(20).std()

    # VWAP (cumulative approx)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    return df


def prepare_features(price_df, delivery_series=None, vix_series=None, fii_dii_df=None):
    """
    Joins auxiliary series (delivery, vix, fii/dii) onto price_df, computes technicals,
    forward/backfill small gaps, and returns cleaned df.
    """
    df = price_df.copy()
    # normalize index to date-only (trading calendar)
    df.index = pd.to_datetime(df.index).date
    df.index = pd.DatetimeIndex(df.index)

    # Join delivery if provided
    if delivery_series is not None:
        s = delivery_series.copy()
        s.index = pd.to_datetime(s.index).date
        s.index = pd.DatetimeIndex(s.index)
        df = df.join(s, how="left")

    # Join VIX
    if vix_series is not None:
        s = vix_series.copy()
        s.index = pd.to_datetime(s.index).date
        s.index = pd.DatetimeIndex(s.index)
        df = df.join(s, how="left")

    # Join FII/DII if provided
    if fii_dii_df is not None:
        f = fii_dii_df.copy()
        if "date" in f.columns:
            f.index = pd.to_datetime(f["date"]).dt.date
            f.index = pd.DatetimeIndex(f.index)
        df = df.join(f[["fii_net", "dii_net"]], how="left")

    df = add_technical_features(df)
    # Fill small gaps and drop remaining NA rows
    df = df.ffill().bfill()
    df = df.dropna()
    return df


# -----------------------------
# Label creation (0..2)
# -----------------------------
def make_labels(df, horizon=1, pct_threshold=0.006):
    """
    Create multiclass labels mapped as:
      0 -> SELL   (future_return < -pct_threshold)
      1 -> HOLD   (abs(future_return) <= pct_threshold)
      2 -> BUY    (future_return > pct_threshold)
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    def lab_to_int(x):
        if x > pct_threshold:
            return 2
        if x < -pct_threshold:
            return 0
        return 1

    df["label"] = df["future_return"].apply(lambda x: lab_to_int(x) if pd.notna(x) else np.nan)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


# -----------------------------
# Model training & prediction
# -----------------------------
def select_feature_columns(df):
    """
    Heuristic: drop raw OHLCV and label/future columns, keep numeric engineered features.
    """
    drop = {"open", "high", "low", "close", "volume", "future_close", "future_return", "label"}
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    # sort for reproducibility
    cols = sorted(cols)
    if not cols:
        raise RuntimeError("No feature columns found after selection.")
    return cols


def train_lightgbm(df, feature_cols):
    X = df[feature_cols]
    y = df["label"]
    # Simple full-training LightGBM (multiclass)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt"
    }
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300)
    return model


def predict_and_format(model, latest_row, feature_cols, stock_name):
    X = latest_row[feature_cols].values.reshape(1, -1)
    probs = model.predict(X)[0]  # shape (3,)
    idx = int(np.argmax(probs))  # 0,1,2
    conf = float(probs[idx])
    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return f"STOCK: {stock_name} — SUGGESTION: {mapping[idx]} — CONFIDENCE: {conf:.2%}"


# -----------------------------
# End-to-end pipeline
# -----------------------------
def run_pipeline(stock_ticker, years=3, pct_threshold=0.006):
    print(f"[info] Fetching price for {stock_ticker} (period={years}y)...")
    price_df = fetch_price_yahoo(stock_ticker, period=f"{years}y", interval="1d")
    if price_df.shape[0] < 60:
        raise RuntimeError("Not enough price data fetched. Increase period or check ticker.")

    start_dt = price_df.index.min()
    end_dt = price_df.index.max()

    print("[info] Attempting to fetch India-specific series (delivery %, India VIX)...")
    delivery = fetch_delivery_jugaad(stock_ticker, start_dt, end_dt) if JUGAAD_OK else None
    if delivery is None and JUGAAD_OK:
        print("[warn] jugaad-data installed but delivery series not found for this symbol.")
    vix = fetch_india_vix(period=f"{years}y")
    if vix is None:
        print("[warn] India VIX not available via yfinance.")

    fii_dii = fetch_fii_dii_stub()  # stub; extend to ingest real FII/DII CSV if available

    print("[info] Preparing features...")
    df_feat = prepare_features(price_df, delivery_series=delivery, vix_series=vix, fii_dii_df=fii_dii)

    print("[info] Creating labels...")
    df_lab = make_labels(df_feat, horizon=1, pct_threshold=pct_threshold)

    feature_cols = select_feature_columns(df_lab)
    print(f"[info] Using {len(feature_cols)} features: {feature_cols}")

    print("[info] Training LightGBM (this may take ~10-30s)...")
    model = train_lightgbm(df_lab, feature_cols)

    latest_row = df_lab.iloc[[-1]]
    suggestion = predict_and_format(model, latest_row, feature_cols, stock_ticker)
    return suggestion


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="India stock suggestion script")
    p.add_argument("--stock", required=True, help="Ticker with exchange suffix (e.g. RELIANCE.NS or TCS.NS)")
    p.add_argument("--years", required=False, default=3, type=int, help="History years to fetch (default 3)")
    p.add_argument("--thr", required=False, default=0.006, type=float, help="Label threshold for BUY/SELL (default 0.006)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        out = run_pipeline(args.stock, years=args.years, pct_threshold=args.thr)
        print(out)
    except Exception as e:
        print("[error]", str(e))
        sys.exit(1)