# app_market_ui.py
"""
Streamlit UI inspired by Grow/Upstox:
- searchable, paginated list of NSE stocks (from NSE archive CSV)
- summary cards with quick heuristic suggestion (BUY/HOLD/SELL)
- per-stock detail view with chart, features, and option to run heavy ML pipeline (if available)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, math, os
from datetime import datetime, timedelta
from functools import partial

# Try to import your heavy pipeline (optional). If present, the app will call run_pipeline(ticker,...)
PIPELINE_AVAILABLE = False
try:
    from stock_suggester_india import run_pipeline  # must return a suggestion string
    PIPELINE_AVAILABLE = True
except Exception:
    PIPELINE_AVAILABLE = False

st.set_page_config(page_title="Market UI — NSE Preview", layout="wide")

# ---------------------------
# Helpers & caching
# ---------------------------
@st.cache_data(ttl=60*60*24)
def fetch_nse_symbols():
    """Fetch NSE EQUITY_L CSV from NSE archives (best-effort)."""
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                df = pd.read_csv(io.StringIO(r.text))
                # normalize columns
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            continue
    return None

@st.cache_data(ttl=60*30)
def batch_fetch_price_info(tickers, period="1y"):
    """
    Use yfinance.download to fetch OHLCV for multiple tickers.
    Returns a dict of dataframes keyed by ticker.
    """
    if not tickers:
        return {}
    # yfinance accepts list without .NS appended; we pass the tickers as provided (with .NS)
    # use period short to be responsive
    try:
        yfdf = yf.download(tickers, period=period, interval="1d", group_by='ticker', auto_adjust=False, threads=True)
    except Exception:
        return {}
    out = {}
    # if single ticker, structure is different
    if isinstance(yfdf.columns, pd.MultiIndex):
        for tk in tickers:
            if (tk,) in yfdf.columns or tk in yfdf.columns.levels[0]:
                try:
                    df = yfdf[tk].dropna()
                    if 'Close' not in df.columns:
                        continue
                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume":"volume"})
                    out[tk] = df[['open','high','low','close','volume']].copy()
                except Exception:
                    continue
    else:
        # single or failed multi: try to build per ticker
        for tk in tickers:
            try:
                df = yf.download(tk, period=period, interval="1d", auto_adjust=False)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                out[tk] = df[['open','high','low','close','volume']].copy()
            except Exception:
                continue
    return out

def compute_quick_features(df):
    """Compute a few quick features for display and heuristic suggestion."""
    d = df.copy()
    d['close'] = d['close'].astype(float)
    d['sma_5'] = d['close'].rolling(5).mean()
    d['sma_20'] = d['close'].rolling(20).mean()
    # RSI simple implementation
    delta = d['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta).clip(lower=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    d['rsi_14'] = 100 - (100 / (1 + rs))
    return d

def quick_suggestion_from_features(latest_row, delivery_pct=None, thr=0.006):
    """
    Fast heuristic:
      if close > sma_5 and close > sma_20 and rsi between (30,70) -> BUY
      if close < sma_20 -> SELL
      else HOLD
    confidence heuristic based on distance from sma_20 and rsi agreement.
    """
    c = latest_row['close']
    sma5 = latest_row.get('sma_5', np.nan)
    sma20 = latest_row.get('sma_20', np.nan)
    rsi = latest_row.get('rsi_14', np.nan)

    # base score
    score = 0.0
    # price vs sma20
    if np.isfinite(sma20):
        diff_pct = (c - sma20) / sma20
        score += np.clip(diff_pct * 5, -1.0, 1.0)  # scale
    # sma5 momentum
    if np.isfinite(sma5):
        score += 0.3 if c > sma5 else -0.3
    # rsi moderation
    if np.isfinite(rsi):
        if rsi < 30:
            score += 0.5
        elif rsi > 70:
            score -= 0.5
    # delivery (if high deliverable% => stronger signal)
    if delivery_pct is not None and not np.isnan(delivery_pct):
        if delivery_pct > 50:
            score += 0.2
        elif delivery_pct < 30:
            score -= 0.1

    # map score to suggestion
    # thresholds tuned for quick UX, not trading
    if score > 0.35:
        suggestion = "BUY"
    elif score < -0.2:
        suggestion = "SELL"
    else:
        suggestion = "HOLD"

    # confidence as logistic on absolute score
    conf = 1/(1 + math.exp(-abs(score)*3))  # in (0.5, 1)
    return suggestion, conf

# ---------------------------
# Layout & interactions
# ---------------------------
st.title("Market UI — NSE-style list (demo)")
st.markdown("Search NSE symbols, view quick summaries, and get suggestions. "
            "Click a stock card to see details and optionally run the full ML pipeline (if available).")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    q = st.text_input("Search symbol / company:", value="")
    per_page = st.selectbox("Cards per page", options=[6, 12, 24, 48], index=1)
    sort_by = st.selectbox("Sort by", options=["Ticker","Price (desc)","1d % (desc)"], index=0)
    show_only = st.selectbox("Show", options=["All","Top Movers (1d)","Top Gainers (1d)","Top Losers (1d)"], index=0)
    refresh = st.button("Refresh symbol list")

# Load symbols
symbols_df = fetch_nse_symbols()
if symbols_df is None:
    st.warning("Could not fetch NSE symbols from archive. You can still type tickers manually (e.g., TCS.NS).")
    # minimal placeholder
    symbols_list = []
else:
    # try to find symbol/name columns
    sym_col = next((c for c in symbols_df.columns if 'symbol' in c.lower()), symbols_df.columns[0])
    name_col = next((c for c in symbols_df.columns if 'name' in c.lower()), None)
    symbols_list = symbols_df[sym_col].astype(str).tolist()

# Build the list of tickers to display by search
if q.strip():
    # search both symbol and company name if available
    qlower = q.strip().lower()
    if symbols_df is not None and name_col is not None:
        filtered_df = symbols_df[symbols_df.apply(lambda row: qlower in str(row[sym_col]).lower() or qlower in str(row[name_col]).lower(), axis=1)]
        tickers = (filtered_df[sym_col].astype(str) + ".NS").tolist()
    elif symbols_df is not None:
        filtered_df = symbols_df[symbols_df[sym_col].astype(str).str.lower().str.contains(qlower)]
        tickers = (filtered_df[sym_col].astype(str) + ".NS").tolist()
    else:
        # fallback: user typed symbol-like string
        tickers = [q.strip().upper()] if '.' in q or len(q.strip())>1 else []
else:
    # default: show first 200 symbols
    tickers = [(s + ".NS") for s in symbols_list[:200]] if symbols_list else []

# add manual entry if not present
manual = st.text_input("Or enter a ticker (e.g., RELIANCE.NS):", value="")
if manual.strip():
    m = manual.strip().upper()
    if m not in tickers:
        tickers = [m] + tickers

# Cap number of fetches to keep app snappy
max_fetch = 60
if len(tickers) > max_fetch:
    st.info(f"Showing first {max_fetch} results (refine search to see more).")
    tickers = tickers[:max_fetch]

# Fetch batch price info (cached)
with st.spinner("Fetching price data..."):
    price_map = batch_fetch_price_info(tickers, period="1mo")  # 1 month enough for summary

# Build summary objects for cards
cards = []
for tk in tickers:
    if tk in price_map:
        df = price_map[tk].dropna()
        if df.empty: continue
        df = df.sort_index()
        dff = compute_quick_features(df)
        latest = dff.iloc[-1]
        prev = dff.iloc[-2] if len(dff) > 1 else latest
        pct1d = (latest['close'] - prev['close'])/prev['close'] if prev['close']!=0 else 0.0
        # 7-day change
        if len(dff) >= 6:
            prev7 = dff['close'].iloc[-6]
            pct7 = (latest['close'] - prev7)/prev7 if prev7!=0 else 0.0
        else:
            pct7 = np.nan
        # delivery placeholder: we don't fetch by default here; could be added
        delivery_pct = None
        suggestion, conf = quick_suggestion_from_features(latest, delivery_pct=delivery_pct)
        cards.append({
            'ticker': tk,
            'close': float(latest['close']),
            'pct1d': float(pct1d),
            'pct7': float(pct7) if not pd.isna(pct7) else None,
            'sma20': float(latest.get('sma_20', np.nan)) if not pd.isna(latest.get('sma_20', np.nan)) else None,
            'rsi': float(latest.get('rsi_14', np.nan)) if not pd.isna(latest.get('rsi_14', np.nan)) else None,
            'suggestion': suggestion,
            'conf': conf,
            'df': dff
        })
    else:
        # no price data (skip or show placeholder)
        cards.append({'ticker': tk, 'close': None, 'pct1d': None, 'pct7': None, 'suggestion': 'N/A', 'conf': 0.0, 'df': None})

# Apply sorting/filtering
if sort_by == "Ticker":
    cards = sorted(cards, key=lambda x: x['ticker'])
elif sort_by == "Price (desc)":
    cards = sorted(cards, key=lambda x: (x['close'] is not None, x['close']), reverse=True)
elif sort_by == "1d % (desc)":
    cards = sorted(cards, key=lambda x: (x['pct1d'] is not None, x['pct1d']), reverse=True)

if show_only == "Top Movers (1d)":
    cards = sorted(cards, key=lambda x: abs(x['pct1d']) if x['pct1d'] is not None else 0.0, reverse=True)
elif show_only == "Top Gainers (1d)":
    cards = sorted(cards, key=lambda x: x['pct1d'] if x['pct1d'] is not None else -9999, reverse=True)
elif show_only == "Top Losers (1d)":
    cards = sorted(cards, key=lambda x: x['pct1d'] if x['pct1d'] is not None else 9999)

# Pagination
total = len(cards)
pages = math.ceil(total / per_page)
page = st.number_input("Page", min_value=1, max_value=max(1,pages), value=1, step=1)
start = (page-1)*per_page
end = start + per_page
page_cards = cards[start:end]

# Render cards in a grid
cols = st.columns(3)
col_index = 0
for card in page_cards:
    with cols[col_index]:
        st.markdown("---")
        # header row with ticker and suggestion badge
        ticker = card['ticker']
        sug = card['suggestion']
        conf_txt = f"{card['conf']:.0%}" if card['conf'] is not None else ""
        st.write(f"**{ticker}** — {sug} {conf_txt}")
        if card['close'] is not None:
            # price and percent
            pct = card['pct1d']*100
            arrow = "🔺" if pct>0 else ("🔻" if pct<0 else "—")
            st.metric(label="Price", value=f"{card['close']:.2f}", delta=f"{pct:.2f}% {arrow}")
            # small line chart
            dff = card['df']
            if dff is not None:
                st.line_chart(dff['close'].tail(30))
                st.write(f"SMA20: {card['sma20']:.2f} | RSI: {card['rsi']:.1f}" if card['sma20'] else "")
        else:
            st.write("No price data")

        # actions: View details, Suggest (ML), Add watch (placeholder)
        cols_actions = st.columns([1,1,1])
        if cols_actions[0].button("View", key=f"view_{ticker}"):
            st.session_state[f"view_{ticker}"] = not st.session_state.get(f"view_{ticker}", False)
        if cols_actions[1].button("Suggest (ML)", key=f"ml_{ticker}"):
            # ML-heavy suggestion (calls user pipeline if available)
            if PIPELINE_AVAILABLE:
                with st.spinner("Running ML pipeline..."):
                    try:
                        result = run_pipeline(ticker, years=3, pct_threshold=0.006)
                        st.success(result)
                    except Exception as e:
                        st.error(f"ML pipeline error: {e}")
            else:
                st.info("ML pipeline not available. Place your stock_suggester_india.py with run_pipeline() in this folder to enable.")

        if cols_actions[2].button("Add Watch", key=f"watch_{ticker}"):
            st.success(f"Added {ticker} to watchlist (demo)")

        # optional detail expander (persistent)
        if st.session_state.get(f"view_{ticker}", False):
            st.expander(f"Details — {ticker}", expanded=True)
            with st.expander(f"Details — {ticker}", expanded=True):
                if card['df'] is not None:
                    st.line_chart(card['df']['close'])
                    st.dataframe(card['df'].tail(20).reset_index().rename(columns={'index':'date'}))
                else:
                    st.write("No detailed data available.")

    col_index = (col_index + 1) % len(cols)

# footer / info
st.markdown("---")
st.caption("This demo uses a fast heuristic to produce suggestions for the list. "
           "You can plug in your heavy ML pipeline by adding stock_suggester_india.py with run_pipeline(stock_ticker, years, pct_threshold).")

