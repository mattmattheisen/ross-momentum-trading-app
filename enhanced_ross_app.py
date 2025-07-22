"""
Ross-Style Momentum Signal Generator (Working Version)
====================================================
Professional momentum trading signals with real-time data.
Compatible with Streamlit Cloud deployment.

Features:
- Full market scanning (8,000+ symbols via Alpaca API)
- Custom symbol list scanning (via Yahoo Finance)
- Multiple Ross Cameron trading strategies
- Real-time signal generation and analysis
- Professional risk management tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time as time_module

# ------------- CONFIGURATION -------------
DEFAULTS = {
    "gap_min": 0.06,            # 6%
    "gap_max": 0.50,            # 50%
    "price_min": 2.0,
    "price_max": 20.0,
    "float_max": 50_000_000,
    "rvol_min": 2.0,
    "pm_vol_min": 150_000,
    "risk_per_trade": 0.005,    # 0.5%
    "max_daily_loss": 0.02,     # 2%
    "stop_buffer": 0.02,
    "target_R": 2.0,
}

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PREMARKET_START = time(4, 0)
PREMARKET_END = time(9, 29)

# ------------- DATA CLASSES -------------
@dataclass
class Signal:
    symbol: str
    setup: str
    entry: float
    stop: float
    target: float
    size: int
    time: str
    notes: str
    confidence: float = 0.0
    risk_reward: float = 0.0
    gap_percent: float = 0.0
    rvol: float = 0.0
    pm_high: float = 0.0
    pm_volume: int = 0

@dataclass
class ScanResult:
    symbol: str
    price: float
    gap_percent: float
    rvol: float
    volume: int
    pm_high: float
    pm_volume: int
    score: int

# ------------- ALPACA CLIENT FUNCTIONS -------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_alpaca_client() -> Optional[StockBarsClient]:
    """Get Alpaca data client with error handling"""
    if not ALPACA_AVAILABLE:
        return None
    
    try:
        key = st.secrets.get("ALPACA_API_KEY", "")
        secret = st.secrets.get("ALPACA_API_SECRET", "")
        if not key or not secret:
            return None
        return StockBarsClient(api_key=key, secret_key=secret)
    except Exception as e:
        st.error(f"Error connecting to Alpaca: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_active_symbols():
    """Get all active US equity symbols from Alpaca"""
    if not ALPACA_AVAILABLE:
        return []
    
    try:
        key = st.secrets.get("ALPACA_API_KEY", "")
        secret = st.secrets.get("ALPACA_API_SECRET", "")
        if not key or not secret:
            return []
        
        tc = TradingClient(key, secret)
        req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = tc.get_all_assets(req)
        
        # NYSE/NASDAQ/AMEX only, skip ETFs and other exchanges
        symbols = [a.symbol for a in assets if a.exchange in ("NYSE", "NASDAQ", "AMEX")]
        return symbols
        
    except Exception as e:
        st.error(f"Error fetching active symbols: {e}")
        return []

def chunk(lst, n):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_daily_bars_alpaca(symbols: List[str], lookback: int, end_date: date, client) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars using Alpaca client"""
    if not client:
        return {}
    
    start = end_date - timedelta(days=lookback*2)
    start_dt = datetime.combine(start, time(4, 0))
    end_dt = datetime.combine(end_date, time(20, 0))
    
    try:
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                               start=start_dt, end=end_dt, adjustment=None)
        data = client.get_stock_bars(req)
        
        out = {}
        for sym, barset in data.data.items():
            df = barset.df
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
            out[sym] = df
        return out
        
    except Exception as e:
        st.error(f"Error fetching daily bars: {e}")
        return {}

def calc_rvol_simple(today_vol: float, hist_volumes: pd.Series, lookback: int = 50) -> float:
    """Calculate relative volume - simple version for scanning"""
    if hist_volumes.empty:
        return np.nan
    avg = hist_volumes.tail(lookback).mean()
    return today_vol / avg if avg > 0 else np.nan

# ------------- MARKET DATA FUNCTIONS -------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol: str, period: str = "5d", interval: str = "1m") -> pd.DataFrame:
    """Get stock data using yfinance with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        # Clean up the dataframe
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Ensure we have the required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            st.warning(f"Missing data columns for {symbol}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  # Cache daily data for 10 minutes
def get_daily_data(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Get daily data for gap and RVOL calculations"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        return df
        
    except Exception as e:
        return pd.DataFrame()

def calculate_gap_percentage(symbol: str) -> float:
    """Calculate gap percentage from previous close"""
    daily_df = get_daily_data(symbol, period="5d")
    
    if len(daily_df) < 2:
        return 0.0
    
    today_open = daily_df.iloc[-1]['open']
    yesterday_close = daily_df.iloc[-2]['close']
    
    if yesterday_close == 0:
        return 0.0
    
    return (today_open - yesterday_close) / yesterday_close

def calculate_rvol(symbol: str, current_volume: float) -> float:
    """Calculate relative volume"""
    daily_df = get_daily_data(symbol, period="50d")
    
    if daily_df.empty or len(daily_df) < 10:
        return 1.0
    
    avg_volume = daily_df['volume'].tail(20).mean()
    
    if avg_volume == 0:
        return 1.0
    
    return current_volume / avg_volume

def get_premarket_data(df: pd.DataFrame) -> Dict:
    """Extract premarket trading data"""
    if df.empty:
        return {"pm_high": 0, "pm_low": 0, "pm_volume": 0, "pm_range": 0}
    
    # Convert datetime column if needed
    if 'datetime' in df.columns:
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['datetime']).dt.time
        
        # Filter premarket hours (4:00 AM - 9:29 AM ET)
        pm_mask = (df_copy['time'] >= PREMARKET_START) & (df_copy['time'] <= PREMARKET_END)
        pm_data = df_copy[pm_mask]
        
        if pm_data.empty:
            # If no exact premarket data, use first 30 minutes of available data
            pm_data = df_copy.head(30)
        
        if not pm_data.empty:
            return {
                "pm_high": pm_data['high'].max(),
                "pm_low": pm_data['low'].min(),
                "pm_volume": pm_data['volume'].sum(),
                "pm_range": pm_data['high'].max() - pm_data['low'].min()
            }
    
    return {"pm_high": 0, "pm_low": 0, "pm_volume": 0, "pm_range": 0}

# ------------- CALCULATION FUNCTIONS -------------
def position_size(entry: float, stop: float, account_size: float, risk_pct: float) -> int:
    """Calculate position size based on risk"""
    if entry <= 0 or stop <= 0 or entry == stop:
        return 0
    
    risk_per_share = abs(entry - stop)
    dollars_at_risk = account_size * risk_pct
    
    if risk_per_share == 0:
        return 0
    
    size = int(dollars_at_risk / risk_per_share)
    return max(1, size)

def calculate_confidence_score(signal_data: dict) -> float:
    """Calculate confidence score for signal quality"""
    score = 0.5  # Base score
    
    gap_pct = signal_data.get('gap_percent', 0)
    rvol = signal_data.get('rvol', 0)
    pm_volume = signal_data.get('pm_volume', 0)
    
    # Gap quality (6-20% is optimal)
    if 0.06 <= abs(gap_pct) <= 0.20:
        score += 0.25
    elif 0.20 < abs(gap_pct) <= 0.35:
        score += 0.15
    elif abs(gap_pct) > 0.50:
        score -= 0.1
    
    # RVOL factor
    if rvol >= 3.0:
        score += 0.20
    elif rvol >= 2.0:
        score += 0.10
    
    # Volume factor
    if pm_volume >= 500_000:
        score += 0.15
    elif pm_volume >= 250_000:
        score += 0.10
    
    return min(1.0, max(0.0, score))

# ------------- STRATEGY IMPLEMENTATIONS -------------
def one_minute_orb_strategy(df: pd.DataFrame, symbol: str, cfg: dict) -> List[Signal]:
    """1-Minute Opening Range Breakout Strategy"""
    signals = []
    
    if df.empty or len(df) < 10:
        return signals
    
    try:
        # Convert datetime and filter for market hours
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['datetime']).dt.time
        
        # Get regular trading hours data
        market_hours = df_copy[df_copy['time'] >= MARKET_OPEN]
        
        if market_hours.empty:
            return signals
        
        # First candle of regular session
        first_candle = market_hours.iloc[0]
        orb_high = first_candle['high']
        orb_low = first_candle['low']
        
        # Look for breakout above ORB high
        breakout_candles = market_hours[market_hours['high'] > orb_high]
        
        if not breakout_candles.empty:
            breakout_time = breakout_candles.iloc[0]['datetime']
            entry = orb_high + 0.01
            stop = orb_low - cfg['stop_buffer']
            target = entry + (entry - stop) * cfg['target_R']
            
            # Get additional data for confidence scoring
            gap_pct = calculate_gap_percentage(symbol)
            current_volume = df['volume'].sum()
            rvol = calculate_rvol(symbol, current_volume)
            pm_data = get_premarket_data(df)
            
            signal_data = {
                'gap_percent': gap_pct,
                'rvol': rvol,
                'pm_volume': pm_data['pm_volume']
            }
            
            size = position_size(entry, stop, cfg.get('account_size', 100000), cfg['risk_per_trade'])
            
            if size > 0:
                signal = Signal(
                    symbol=symbol,
                    setup="1m_ORB",
                    entry=entry,
                    stop=stop,
                    target=target,
                    size=size,
                    time=str(breakout_time),
                    notes=f"ORB breakout above ${orb_high:.2f}",
                    confidence=calculate_confidence_score(signal_data),
                    risk_reward=cfg['target_R'],
                    gap_percent=gap_pct,
                    rvol=rvol,
                    pm_high=pm_data['pm_high'],
                    pm_volume=pm_data['pm_volume']
                )
                signals.append(signal)
        
    except Exception as e:
        st.error(f"Error in ORB strategy for {symbol}: {str(e)}")
    
    return signals

def premarket_high_break_strategy(df: pd.DataFrame, symbol: str, cfg: dict) -> List[Signal]:
    """Premarket High Breakout Strategy"""
    signals = []
    
    if df.empty:
        return signals
    
    try:
        pm_data = get_premarket_data(df)
        pm_high = pm_data['pm_high']
        
        if pm_high <= 0:
            return signals
        
        # Check for break of premarket high
        df_copy = df.copy()
        df_copy['time'] = pd.to_datetime(df_copy['datetime']).dt.time
        
        # Look in regular trading hours
        market_hours = df_copy[df_copy['time'] >= MARKET_OPEN]
        
        if market_hours.empty:
            return signals
        
        # Find breakout above PM high
        breakout_candles = market_hours[market_hours['high'] > pm_high]
        
        if not breakout_candles.empty:
            breakout_time = breakout_candles.iloc[0]['datetime']
            entry = pm_high + 0.01
            
            # Dynamic stop based on recent lows
            recent_lows = market_hours['low'].head(10)
            if not recent_lows.empty:
                stop = recent_lows.min() - cfg['stop_buffer']
            else:
                stop = entry * 0.97  # 3% fallback stop
            
            target = entry + (entry - stop) * cfg['target_R']
            
            # Additional metrics
            gap_pct = calculate_gap_percentage(symbol)
            current_volume = df['volume'].sum()
            rvol = calculate_rvol(symbol, current_volume)
            
            signal_data = {
                'gap_percent': gap_pct,
                'rvol': rvol,
                'pm_volume': pm_data['pm_volume']
            }
            
            size = position_size(entry, stop, cfg.get('account_size', 100000), cfg['risk_per_trade'])
            
            if size > 0:
                signal = Signal(
                    symbol=symbol,
                    setup="PM_High_Break",
                    entry=entry,
                    stop=stop,
                    target=target,
                    size=size,
                    time=str(breakout_time),
                    notes=f"PM high break ${pm_high:.2f} (Vol: {pm_data['pm_volume']:,})",
                    confidence=calculate_confidence_score(signal_data),
                    risk_reward=cfg['target_R'],
                    gap_percent=gap_pct,
                    rvol=rvol,
                    pm_high=pm_high,
                    pm_volume=pm_data['pm_volume']
                )
                signals.append(signal)
        
    except Exception as e:
        st.error(f"Error in PM high break strategy for {symbol}: {str(e)}")
    
    return signals

def gap_and_go_strategy(df: pd.DataFrame, symbol: str, cfg: dict) -> List[Signal]:
    """Gap and Go Strategy - simplified version"""
    signals = []
    
    if df.empty:
        return signals
    
    try:
        gap_pct = calculate_gap_percentage(symbol)
        
        # Must have significant gap
        if abs(gap_pct) < cfg['gap_min']:
            return signals
        
        current_price = df.iloc[-1]['close']
        pm_data = get_premarket_data(df)
        
        # Entry above current resistance
        entry = current_price + 0.01
        stop = current_price * 0.97  # 3% stop
        target = entry + (entry - stop) * cfg['target_R']
        
        # Additional metrics
        current_volume = df['volume'].sum()
        rvol = calculate_rvol(symbol, current_volume)
        
        signal_data = {
            'gap_percent': gap_pct,
            'rvol': rvol,
            'pm_volume': pm_data['pm_volume']
        }
        
        size = position_size(entry, stop, cfg.get('account_size', 100000), cfg['risk_per_trade'])
        
        if size > 0:
            signal = Signal(
                symbol=symbol,
                setup="Gap_and_Go",
                entry=entry,
                stop=stop,
                target=target,
                size=size,
                time=str(df.iloc[-1]['datetime']),
                notes=f"Gap {gap_pct:.1%} momentum play",
                confidence=calculate_confidence_score(signal_data),
                risk_reward=cfg['target_R'],
                gap_percent=gap_pct,
                rvol=rvol,
                pm_high=pm_data['pm_high'],
                pm_volume=pm_data['pm_volume']
            )
            signals.append(signal)
        
    except Exception as e:
        st.error(f"Error in Gap and Go strategy for {symbol}: {str(e)}")
    
    return signals

# ------------- SCANNING FUNCTIONS -------------
def scan_for_opportunities(symbols: List[str], cfg: dict) -> pd.DataFrame:
    """Scan symbols for trading opportunities"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        status_text.text(f"Scanning {symbol}...")
        
        try:
            # Get current price
            daily_df = get_daily_data(symbol, period="5d")
            if daily_df.empty:
                continue
            
            current_price = daily_df.iloc[-1]['close']
            
            # Apply basic filters
            if not (cfg['price_min'] <= current_price <= cfg['price_max']):
                continue
            
            # Calculate gap
            gap_pct = calculate_gap_percentage(symbol)
            if abs(gap_pct) < cfg['gap_min'] or abs(gap_pct) > cfg.get('gap_max', 1.0):
                continue
            
            # Get intraday data
            intraday_df = get_stock_data(symbol, period="1d", interval="1m")
            pm_data = get_premarket_data(intraday_df)
            
            # Calculate RVOL
            current_volume = daily_df.iloc[-1]['volume']
            rvol = calculate_rvol(symbol, current_volume)
            
            if rvol >= cfg['rvol_min']:
                score = calculate_opportunity_score(gap_pct, rvol, current_price, pm_data)
                
                results.append({
                    'Symbol': symbol,
                    'Price': f"${current_price:.2f}",
                    'Gap %': f"{gap_pct:.1%}",
                    'RVOL': f"{rvol:.1f}",
                    'Volume': f"{int(current_volume):,}",
                    'PM High': f"${pm_data['pm_high']:.2f}" if pm_data['pm_high'] > 0 else "N/A",
                    'PM Vol': f"{int(pm_data['pm_volume']):,}" if pm_data['pm_volume'] > 0 else "N/A",
                    'Score': score
                })
        
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    df_results = pd.DataFrame(results)
    return df_results.sort_values('Score', ascending=False) if not df_results.empty else df_results

def calculate_opportunity_score(gap_pct: float, rvol: float, price: float, pm_data: dict) -> int:
    """Calculate opportunity score for ranking"""
    score = 0
    
    # Gap score (6-20% optimal)
    if 0.06 <= abs(gap_pct) <= 0.20:
        score += 40
    elif 0.20 < abs(gap_pct) <= 0.35:
        score += 25
    elif abs(gap_pct) > 0.50:
        score += 10
    
    # RVOL score
    if rvol >= 5:
        score += 30
    elif rvol >= 3:
        score += 20
    elif rvol >= 2:
        score += 10
    
    # Price sweet spot
    if 3 <= price <= 15:
        score += 20
    elif 2 <= price <= 20:
        score += 10
    
    # PM volume
    pm_vol = pm_data.get('pm_volume', 0)
    if pm_vol >= 500_000:
        score += 10
    elif pm_vol >= 250_000:
        score += 5
    
    return score

# ------------- VISUALIZATION -------------
def create_trading_chart(df: pd.DataFrame, signals: List[Signal], symbol: str) -> go.Figure:
    """Create comprehensive trading chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Price Action', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add signals
    for signal in signals:
        if signal.symbol == symbol:
            signal_time = pd.to_datetime(signal.time)
            
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[signal_time],
                    y=[signal.entry],
                    mode='markers',
                    marker=dict(color='green', size=12, symbol='triangle-up'),
                    name=f'{signal.setup} Entry',
                    hovertemplate=f'<b>{signal.setup}</b><br>Entry: ${signal.entry:.2f}<br>Stop: ${signal.stop:.2f}<br>Target: ${signal.target:.2f}<br>Size: {signal.size}<br>Confidence: {signal.confidence:.0%}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Stop and target lines
            fig.add_hline(y=signal.stop, line_dash="dash", line_color="red", 
                         annotation_text=f"Stop: ${signal.stop:.2f}", row=1, col=1)
            fig.add_hline(y=signal.target, line_dash="dash", line_color="green",
                         annotation_text=f"Target: ${signal.target:.2f}", row=1, col=1)
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Market session backgrounds
    if not df.empty:
        market_open_time = pd.to_datetime(df.iloc[0]['datetime']).replace(hour=9, minute=30, second=0)
        
        # Premarket background
        fig.add_vrect(
            x0=df.iloc[0]['datetime'],
            x1=market_open_time,
            fillcolor="yellow",
            opacity=0.1,
            annotation_text="Premarket",
            annotation_position="top left",
            row=1, col=1
        )
    
    fig.update_layout(
        title=f"{symbol} Trading Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

# ------------- STREAMLIT APP -------------
def sidebar_configuration() -> dict:
    """Create sidebar configuration"""
    st.sidebar.header("🎯 Trading Configuration")
    
    # Market filters
    with st.sidebar.expander("📈 Market Filters", expanded=True):
        gap_min = st.slider("Gap % Min", 0.01, 0.50, DEFAULTS['gap_min'], 0.01)
        gap_max = st.slider("Gap % Max", 0.10, 1.00, 0.50, 0.05)
        price_range = st.slider("Price Range ($)", 1.0, 100.0, (DEFAULTS['price_min'], DEFAULTS['price_max']))
        rvol_min = st.slider("RVOL Min", 0.5, 10.0, DEFAULTS['rvol_min'], 0.1)
    
    # Risk management
    with st.sidebar.expander("⚠️ Risk Management", expanded=True):
        account_size = st.number_input("Account Size ($)", 1000, 10000000, 100000, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 3.0, 0.5) / 100
        target_r = st.slider("Target R Multiple", 1.0, 5.0, DEFAULTS['target_R'], 0.25)
        stop_buffer = st.number_input("Stop Buffer ($)", 0.01, 0.50, DEFAULTS['stop_buffer'], 0.01)
    
    return {
        'gap_min': gap_min,
        'gap_max': gap_max,
        'price_min': price_range[0],
        'price_max': price_range[1],
        'rvol_min': rvol_min,
        'account_size': account_size,
        'risk_per_trade': risk_per_trade,
        'target_R': target_r,
        'stop_buffer': stop_buffer,
        'pm_vol_min': DEFAULTS['pm_vol_min']
    }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Ross-Style Momentum Signals",
        page_icon="🚀",
        layout="wide"
    )
    
    # Header
    st.title("🚀 Ross-Style Momentum Trading Signals")
    st.caption("Professional momentum trading signal generator • Educational use only")
    
    # Sidebar configuration
    cfg = sidebar_configuration()
    
    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        trade_date = st.date_input("Trading Date", value=date.today())
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        st.metric("Market", "OPEN" if datetime.now().time() < MARKET_CLOSE else "CLOSED")
    
    # Symbol input
    st.subheader("🎯 Trading Universe")
    
    # Popular momentum stocks
    popular_stocks = [
        "AAPL", "TSLA", "NVDA", "AMD", "SOUN", "IONQ", "AI", "PLTR", "SMCI", "AVGO",
        "GOOGL", "MSFT", "META", "AMZN", "NFLX", "CRM", "ADBE", "ORCL", "UBER", "SHOP",
        "COIN", "ROKU", "ZM", "PTON", "SNAP", "SQ", "PYPL", "ARKK", "QQQ", "SPY"
    ]
    
    # Check if we have market scan results to suggest
    suggested_symbols = popular_stocks[:15]
    if 'market_scan_results' in st.session_state and not st.session_state.market_scan_results.empty:
        # Use top symbols from market scan
        top_market_symbols = st.session_state.market_scan_results.head(15)['Symbol'].str.replace('
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Market Scanner",
        "📈 Signal Generation", 
        "📊 Trading Charts",
        "⚠️ Risk Management",
        "ℹ️ Information"
    ])
    
    with tab1:
        st.subheader("🔍 Market Opportunity Scanner")
        
        # Two scanner options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Option 1: Custom Symbol List**")
            if st.button("🚀 Run Custom Scan", type="primary"):
                if symbols:
                    with st.spinner("Scanning your custom symbols..."):
                        scan_results = scan_for_opportunities(symbols, cfg)
                    
                    if not scan_results.empty:
                        st.success(f"🎯 Found **{len(scan_results)}** opportunities!")
                        
                        # Display results
                        st.dataframe(
                            scan_results,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Opportunity Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        # Download option
                        csv = scan_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Custom Scan",
                            data=csv,
                            file_name=f"custom_scan_{trade_date}.csv",
                            mime="text/csv"
                        )
                        
                        # Store results for other tabs
                        st.session_state.scan_results = scan_results
                        
                    else:
                        st.warning("No opportunities found in your symbol list.")
                else:
                    st.error("Please enter symbols to scan.")
        
        with col2:
            st.write("**Option 2: Full Market Scan (Requires Alpaca API)**")
            alpaca_client = get_alpaca_client()
            
            if alpaca_client and ALPACA_AVAILABLE:
                if st.button("🌍 Scan Entire Market", type="secondary"):
                    with st.spinner("Scanning entire market... may take 30-60 seconds"):
                        df_scan = full_market_gap_scan(trade_date, cfg, alpaca_client)
                    
                    if not df_scan.empty:
                        st.success(f"🎯 Found **{len(df_scan)}** market opportunities!")
                        
                        # Display top 100 results
                        display_df = df_scan.head(100)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        st.download_button(
                            "📥 Download Full Market Scan", 
                            df_scan.to_csv(index=False), 
                            f"full_market_scan_{trade_date}.csv"
                        )
                        
                        st.caption("💡 Tip: After scanning, copy symbols from results to use in the Signals tab.")
                        
                        # Store full market results
                        st.session_state.market_scan_results = df_scan
                        
                        # Quick copy feature for top symbols
                        top_symbols = df_scan.head(20)['Symbol'].tolist()
                        top_symbols_str = ",".join(top_symbols)
                        
                        st.text_area(
                            "🔥 Top 20 Symbols (copy for Signal Generation):",
                            value=top_symbols_str,
                            height=80,
                            help="Copy these symbols and paste into the Signal Generation tab"
                        )
                        
                    else:
                        st.warning("No opportunities found in full market scan. Try adjusting your filters.")
            else:
                st.info("""
                **Full market scanning requires Alpaca API credentials.**
                
                To enable:
                1. Get free Alpaca account at alpaca.markets
                2. Add your API keys to Streamlit secrets
                3. Refresh this app
                
                **Benefits of full market scan:**
                - Scans 8,000+ NYSE/NASDAQ/AMEX stocks automatically
                - Finds opportunities you might miss manually
                - Ranks all opportunities by momentum potential
                - Updates with real-time market data
                """)
                
                # Show how to add API keys
                with st.expander("📝 How to Add Alpaca API Keys"):
                    st.code("""
                    # Add to your Streamlit app secrets:
                    ALPACA_API_KEY = "your_api_key_here"
                    ALPACA_API_SECRET = "your_secret_key_here"
                    """)
        
        # Results summary
        if 'scan_results' in st.session_state or 'market_scan_results' in st.session_state:
            st.divider()
            st.write("### 📊 Scanner Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'scan_results' in st.session_state:
                    custom_count = len(st.session_state.scan_results)
                    st.metric("Custom Scan Results", custom_count)
            
            with col2:
                if 'market_scan_results' in st.session_state:
                    market_count = len(st.session_state.market_scan_results)
                    st.metric("Full Market Results", market_count)
    
    with tab2:
        st.subheader("📈 Trading Signal Generation")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        with col1:
            strategies = st.multiselect(
                "Select Trading Strategies:",
                ["1m_ORB", "PM_High_Break", "Gap_and_Go"],
                default=["1m_ORB", "PM_High_Break"],
                help="Choose which momentum strategies to apply"
            )
        
        with col2:
            max_signals = st.slider("Max Signals per Symbol", 1, 5, 2)
        
        # Symbol selection for signals
        available_symbols = symbols[:10]  # Limit for performance
        selected_symbols = st.multiselect(
            "Select symbols for signal generation:",
            available_symbols,
            default=available_symbols[:5],
            help="Select up to 10 symbols for detailed signal analysis"
        )
        
        if st.button("🎯 Generate Trading Signals", type="primary"):
            if selected_symbols and strategies:
                all_signals = []
                
                progress_bar = st.progress(0)
                
                for i, symbol in enumerate(selected_symbols):
                    progress_bar.progress((i + 1) / len(selected_symbols))
                    
                    # Get intraday data
                    df = get_stock_data(symbol, period="1d", interval="1m")
                    
                    if df.empty:
                        continue
                    
                    # Apply selected strategies
                    if "1m_ORB" in strategies:
                        orb_signals = one_minute_orb_strategy(df, symbol, cfg)
                        all_signals.extend(orb_signals[:max_signals])
                    
                    if "PM_High_Break" in strategies:
                        pm_signals = premarket_high_break_strategy(df, symbol, cfg)
                        all_signals.extend(pm_signals[:max_signals])
                    
                    if "Gap_and_Go" in strategies:
                        gap_signals = gap_and_go_strategy(df, symbol, cfg)
                        all_signals.extend(gap_signals[:max_signals])
                
                progress_bar.empty()
                
                if all_signals:
                    st.success(f"🎯 Generated **{len(all_signals)}** trading signals!")
                    
                    # Convert signals to display format
                    signal_data = []
                    for signal in all_signals:
                        signal_data.append({
                            'Symbol': signal.symbol,
                            'Strategy': signal.setup,
                            'Entry': f"${signal.entry:.2f}",
                            'Stop': f"${signal.stop:.2f}",
                            'Target': f"${signal.target:.2f}",
                            'Size': f"{signal.size:,}",
                            'Risk': f"${abs(signal.entry - signal.stop) * signal.size:.0f}",
                            'Reward': f"${abs(signal.target - signal.entry) * signal.size:.0f}",
                            'R:R': f"{signal.risk_reward:.1f}R",
                            'Confidence': f"{signal.confidence:.0%}",
                            'Gap %': f"{signal.gap_percent:.1%}",
                            'RVOL': f"{signal.rvol:.1f}",
                            'Notes': signal.notes
                        })
                    
                    signals_df = pd.DataFrame(signal_data)
                    
                    # Display signals table
                    st.dataframe(
                        signals_df,
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=1,
                                format="%.0%%"
                            )
                        }
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_risk = sum(abs(s.entry - s.stop) * s.size for s in all_signals)
                    total_reward = sum(abs(s.target - s.entry) * s.size for s in all_signals)
                    avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
                    
                    col1.metric("Total Signals", len(all_signals))
                    col2.metric("Total Risk", f"${total_risk:,.0f}")
                    col3.metric("Potential Reward", f"${total_reward:,.0f}")
                    col4.metric("Avg Confidence", f"{avg_confidence:.0%}")
                    
                    # Download signals
                    signals_csv = pd.DataFrame([asdict(s) for s in all_signals]).to_csv(index=False)
                    st.download_button(
                        "📥 Download Trading Signals",
                        data=signals_csv,
                        file_name=f"trading_signals_{trade_date}.csv",
                        mime="text/csv"
                    )
                    
                    # Store signals for charts tab
                    st.session_state.trading_signals = all_signals
                    st.session_state.signal_symbols = selected_symbols
                    
                else:
                    st.info("No signals generated. Try different symbols or adjust your filters.")
            else:
                st.error("Please select symbols and strategies.")
    
    with tab3:
        st.subheader("📊 Trading Charts & Analysis")
        
        # Check if we have signals to display
        if 'trading_signals' in st.session_state and st.session_state.trading_signals:
            # Symbol selection for charting
            chart_symbols = list(set([s.symbol for s in st.session_state.trading_signals]))
            
            selected_chart_symbol = st.selectbox(
                "Select symbol to analyze:",
                chart_symbols,
                help="Choose a symbol that has generated signals"
            )
            
            if selected_chart_symbol:
                # Get the data and signals for this symbol
                df = get_stock_data(selected_chart_symbol, period="1d", interval="1m")
                symbol_signals = [s for s in st.session_state.trading_signals if s.symbol == selected_chart_symbol]
                
                if not df.empty:
                    # Create and display chart
                    fig = create_trading_chart(df, symbol_signals, selected_chart_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current market data
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    current_price = df.iloc[-1]['close']
                    day_open = df.iloc[0]['open']
                    day_high = df['high'].max()
                    day_low = df['low'].min()
                    day_volume = df['volume'].sum()
                    
                    day_change = current_price - day_open
                    day_change_pct = (day_change / day_open) * 100 if day_open > 0 else 0
                    
                    col1.metric("Current Price", f"${current_price:.2f}", f"{day_change:+.2f} ({day_change_pct:+.1f}%)")
                    col2.metric("Day High", f"${day_high:.2f}")
                    col3.metric("Day Low", f"${day_low:.2f}")
                    col4.metric("Day Volume", f"{day_volume:,.0f}")
                    col5.metric("Range", f"${day_high - day_low:.2f}")
                    
                    # Signal details for this symbol
                    if symbol_signals:
                        st.subheader(f"📈 Active Signals for {selected_chart_symbol}")
                        
                        for signal in symbol_signals:
                            with st.expander(f"{signal.setup} - {signal.confidence:.0%} Confidence"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Entry Details:**")
                                    st.write(f"Entry Price: ${signal.entry:.2f}")
                                    st.write(f"Stop Loss: ${signal.stop:.2f}")
                                    st.write(f"Target: ${signal.target:.2f}")
                                
                                with col2:
                                    st.write("**Position Details:**")
                                    st.write(f"Position Size: {signal.size:,} shares")
                                    st.write(f"Risk per Share: ${abs(signal.entry - signal.stop):.2f}")
                                    st.write(f"Total Risk: ${abs(signal.entry - signal.stop) * signal.size:.0f}")
                                
                                with col3:
                                    st.write("**Signal Quality:**")
                                    st.write(f"Confidence: {signal.confidence:.0%}")
                                    st.write(f"Gap %: {signal.gap_percent:.1%}")
                                    st.write(f"RVOL: {signal.rvol:.1f}x")
                                
                                st.write(f"**Notes:** {signal.notes}")
                else:
                    st.error(f"Could not load chart data for {selected_chart_symbol}")
        else:
            st.info("Generate trading signals first to see charts and analysis.")
            
            # Option to view any symbol without signals
            st.subheader("📈 View Any Symbol")
            manual_symbol = st.text_input("Enter symbol for chart analysis:", "AAPL")
            
            if st.button("Load Chart") and manual_symbol:
                df = get_stock_data(manual_symbol.upper(), period="1d", interval="1m")
                if not df.empty:
                    fig = create_trading_chart(df, [], manual_symbol.upper())
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Risk Management Tools")
        
        # Position size calculator
        st.write("### 🧮 Position Size Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            calc_entry = st.number_input("Entry Price ($)", value=10.00, min_value=0.01, step=0.01)
            calc_stop = st.number_input("Stop Loss ($)", value=9.50, min_value=0.01, step=0.01)
            calc_account = st.number_input("Account Size ($)", value=cfg['account_size'], min_value=1000, step=1000)
            calc_risk = st.slider("Risk per Trade (%)", 0.1, 5.0, cfg['risk_per_trade'] * 100) / 100
        
        with col2:
            if calc_entry > 0 and calc_stop > 0 and calc_entry != calc_stop:
                calc_size = position_size(calc_entry, calc_stop, calc_account, calc_risk)
                risk_per_share = abs(calc_entry - calc_stop)
                total_risk = risk_per_share * calc_size
                total_position_value = calc_entry * calc_size
                
                st.metric("Recommended Size", f"{calc_size:,} shares")
                st.metric("Risk per Share", f"${risk_per_share:.2f}")
                st.metric("Total Risk", f"${total_risk:.0f}")
                st.metric("Position Value", f"${total_position_value:,.0f}")
                
                if total_position_value > calc_account:
                    st.warning("⚠️ Position value exceeds account size!")
            else:
                st.info("Enter valid entry and stop prices to calculate position size.")
        
        # Daily P&L tracker
        st.write("### 📊 Daily P&L Tracker")
        
        # Initialize session state for P&L tracking
        if 'daily_trades' not in st.session_state:
            st.session_state.daily_trades = []
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_pnl = st.number_input("Trade P&L ($)", value=0.0, step=10.0)
            trade_symbol = st.text_input("Symbol (optional)", value="")
        
        with col2:
            if st.button("➕ Add Trade"):
                st.session_state.daily_trades.append({
                    'symbol': trade_symbol or 'N/A',
                    'pnl': trade_pnl,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                st.success(f"Added trade: ${trade_pnl:.2f}")
        
        with col3:
            if st.button("🗑️ Reset Day"):
                st.session_state.daily_trades = []
                st.success("Daily P&L reset")
        
        # Display daily stats
        if st.session_state.daily_trades:
            trades_df = pd.DataFrame(st.session_state.daily_trades)
            
            total_pnl = trades_df['pnl'].sum()
            trade_count = len(trades_df)
            winners = len(trades_df[trades_df['pnl'] > 0])
            losers = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winners / trade_count) * 100 if trade_count > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Daily P&L", f"${total_pnl:.2f}", f"{(total_pnl/cfg['account_size'])*100:.2f}%")
            col2.metric("Total Trades", trade_count)
            col3.metric("Winners", winners)
            col4.metric("Losers", losers)
            col5.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Check daily loss limit
            daily_loss_limit = cfg['account_size'] * cfg.get('max_daily_loss', 0.02)
            if total_pnl < -daily_loss_limit:
                st.error(f"🚨 **DAILY LOSS LIMIT EXCEEDED!** Stop trading. Loss: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            elif total_pnl < -daily_loss_limit * 0.75:
                st.warning(f"⚠️ **Approaching daily loss limit.** Current: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            
            # Show recent trades
            st.write("**Recent Trades:**")
            st.dataframe(trades_df.tail(10), use_container_width=True)
    
    with tab5:
        st.subheader("ℹ️ Trading Information & Disclaimers")
        
        # Strategy explanations
        st.write("### 🎯 Implemented Trading Strategies")
        
        with st.expander("1-Minute Opening Range Breakout (ORB)"):
            st.write("""
            **Strategy:** Break of the first 1-minute candle of regular trading hours.
            
            **Entry:** $0.01 above the high of the first 1-minute candle (9:30-9:31 AM ET)
            **Stop:** Below the low of the first candle minus buffer
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Strong gap-ups with good volume and range in the opening candle.
            """)
        
        with st.expander("Premarket High Break"):
            st.write("""
            **Strategy:** Breakout above the highest price reached during premarket trading.
            
            **Entry:** $0.01 above premarket high
            **Stop:** Below recent structural low minus buffer  
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Stocks with significant premarket activity and volume.
            """)
        
        with st.expander("Gap and Go"):
            st.write("""
            **Strategy:** Momentum continuation after significant overnight gap.
            
            **Entry:** Above current resistance levels
            **Stop:** Percentage-based or technical level
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Clean gaps with strong volume and clear direction.
            """)
        
        # Risk management info
        st.write("### ⚠️ Risk Management Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Position Sizing:**
            - Based on account risk percentage
            - Calculated from entry to stop distance
            - Never exceed account buying power
            - Adjust for volatility and liquidity
            """)
        
        with col2:
            st.write("""
            **Daily Limits:**
            - Maximum risk per trade: 0.5-1% of account
            - Daily loss limit: 2% of account  
            - Stop trading when limit reached
            - Review and adjust rules regularly
            """)
        
        # Important disclaimers
        st.write("### 📋 Important Disclaimers")
        
        st.error("""
        **⚠️ RISK WARNING:**
        - Trading involves substantial risk and is not suitable for all investors
        - Past performance does not guarantee future results  
        - You can lose more than your initial investment
        - Only trade with money you can afford to lose completely
        """)
        
        st.warning("""
        **📚 EDUCATIONAL USE ONLY:**
        - This application is for educational and research purposes
        - Not intended as investment or trading advice
        - Consult qualified financial professionals before trading
        - Practice with paper trading before risking real money
        """)
        
        st.info("""
        **🔧 TECHNICAL NOTES:**
        - Uses Yahoo Finance for free market data (15-20 minute delay)
        - Real-time data requires premium data feeds
        - Signals are based on historical patterns and technical analysis
        - Market conditions can change rapidly - always monitor positions
        """)
        
        # App information
        st.write("### 🚀 About This Application")
        st.write("""
        **Version:** 1.0 - Working Release
        **Data Source:** Yahoo Finance (Free)
        **Update Frequency:** Real-time for display, cached for performance
        **Deployment:** Streamlit Cloud
        
        **Features:**
        - Real-time market scanning
        - Multiple momentum trading strategies
        - Risk-based position sizing
        - Interactive charting and analysis  
        - Daily P&L tracking
        - Professional signal generation
        
        **Recommended Usage:**
        1. Run morning scan before market open
        2. Review highest-scoring opportunities
        3. Generate signals for selected symbols
        4. Analyze charts and confirm setups
        5. Calculate position sizes and risk
        6. Execute trades manually in your broker
        7. Track performance and adjust strategy
        
        ---
        **Remember:** The best traders prioritize risk management over profits. Trade smart, trade safe! 📈
        """)

if __name__ == "__main__":
    main(), '').tolist()
        suggested_symbols = top_market_symbols
        st.info("💡 Showing top symbols from your latest full market scan!")
    
    default_universe = st.text_area(
        "Enter symbols (comma-separated):",
        value=",".join(suggested_symbols),
        height=80,
        help="Enter stock symbols separated by commas. Popular momentum stocks are pre-loaded, or use results from your market scan."
    )
    
    symbols = [s.strip().upper().replace('
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Market Scanner",
        "📈 Signal Generation", 
        "📊 Trading Charts",
        "⚠️ Risk Management",
        "ℹ️ Information"
    ])
    
    with tab1:
        st.subheader("🔍 Market Opportunity Scanner")
        
        # Two scanner options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Option 1: Custom Symbol List**")
            if st.button("🚀 Run Custom Scan", type="primary"):
                if symbols:
                    with st.spinner("Scanning your custom symbols..."):
                        scan_results = scan_for_opportunities(symbols, cfg)
                    
                    if not scan_results.empty:
                        st.success(f"🎯 Found **{len(scan_results)}** opportunities!")
                        
                        # Display results
                        st.dataframe(
                            scan_results,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Opportunity Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        # Download option
                        csv = scan_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Custom Scan",
                            data=csv,
                            file_name=f"custom_scan_{trade_date}.csv",
                            mime="text/csv"
                        )
                        
                        # Store results for other tabs
                        st.session_state.scan_results = scan_results
                        
                    else:
                        st.warning("No opportunities found in your symbol list.")
                else:
                    st.error("Please enter symbols to scan.")
        
        with col2:
            st.write("**Option 2: Full Market Scan (Requires Alpaca API)**")
            alpaca_client = get_alpaca_client()
            
            if alpaca_client and ALPACA_AVAILABLE:
                if st.button("🌍 Scan Entire Market", type="secondary"):
                    with st.spinner("Scanning entire market... may take 30-60 seconds"):
                        df_scan = full_market_gap_scan(trade_date, cfg, alpaca_client)
                    
                    if not df_scan.empty:
                        st.success(f"🎯 Found **{len(df_scan)}** market opportunities!")
                        
                        # Display top 100 results
                        display_df = df_scan.head(100)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        st.download_button(
                            "📥 Download Full Market Scan", 
                            df_scan.to_csv(index=False), 
                            f"full_market_scan_{trade_date}.csv"
                        )
                        
                        st.caption("💡 Tip: After scanning, copy symbols from results to use in the Signals tab.")
                        
                        # Store full market results
                        st.session_state.market_scan_results = df_scan
                        
                        # Quick copy feature for top symbols
                        top_symbols = df_scan.head(20)['Symbol'].tolist()
                        top_symbols_str = ",".join(top_symbols)
                        
                        st.text_area(
                            "🔥 Top 20 Symbols (copy for Signal Generation):",
                            value=top_symbols_str,
                            height=80,
                            help="Copy these symbols and paste into the Signal Generation tab"
                        )
                        
                    else:
                        st.warning("No opportunities found in full market scan. Try adjusting your filters.")
            else:
                st.info("""
                **Full market scanning requires Alpaca API credentials.**
                
                To enable:
                1. Get free Alpaca account at alpaca.markets
                2. Add your API keys to Streamlit secrets
                3. Refresh this app
                
                **Benefits of full market scan:**
                - Scans 8,000+ NYSE/NASDAQ/AMEX stocks automatically
                - Finds opportunities you might miss manually
                - Ranks all opportunities by momentum potential
                - Updates with real-time market data
                """)
                
                # Show how to add API keys
                with st.expander("📝 How to Add Alpaca API Keys"):
                    st.code("""
                    # Add to your Streamlit app secrets:
                    ALPACA_API_KEY = "your_api_key_here"
                    ALPACA_API_SECRET = "your_secret_key_here"
                    """)
        
        # Results summary
        if 'scan_results' in st.session_state or 'market_scan_results' in st.session_state:
            st.divider()
            st.write("### 📊 Scanner Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'scan_results' in st.session_state:
                    custom_count = len(st.session_state.scan_results)
                    st.metric("Custom Scan Results", custom_count)
            
            with col2:
                if 'market_scan_results' in st.session_state:
                    market_count = len(st.session_state.market_scan_results)
                    st.metric("Full Market Results", market_count)
    
    with tab2:
        st.subheader("📈 Trading Signal Generation")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        with col1:
            strategies = st.multiselect(
                "Select Trading Strategies:",
                ["1m_ORB", "PM_High_Break", "Gap_and_Go"],
                default=["1m_ORB", "PM_High_Break"],
                help="Choose which momentum strategies to apply"
            )
        
        with col2:
            max_signals = st.slider("Max Signals per Symbol", 1, 5, 2)
        
        # Symbol selection for signals
        available_symbols = symbols[:10]  # Limit for performance
        selected_symbols = st.multiselect(
            "Select symbols for signal generation:",
            available_symbols,
            default=available_symbols[:5],
            help="Select up to 10 symbols for detailed signal analysis"
        )
        
        if st.button("🎯 Generate Trading Signals", type="primary"):
            if selected_symbols and strategies:
                all_signals = []
                
                progress_bar = st.progress(0)
                
                for i, symbol in enumerate(selected_symbols):
                    progress_bar.progress((i + 1) / len(selected_symbols))
                    
                    # Get intraday data
                    df = get_stock_data(symbol, period="1d", interval="1m")
                    
                    if df.empty:
                        continue
                    
                    # Apply selected strategies
                    if "1m_ORB" in strategies:
                        orb_signals = one_minute_orb_strategy(df, symbol, cfg)
                        all_signals.extend(orb_signals[:max_signals])
                    
                    if "PM_High_Break" in strategies:
                        pm_signals = premarket_high_break_strategy(df, symbol, cfg)
                        all_signals.extend(pm_signals[:max_signals])
                    
                    if "Gap_and_Go" in strategies:
                        gap_signals = gap_and_go_strategy(df, symbol, cfg)
                        all_signals.extend(gap_signals[:max_signals])
                
                progress_bar.empty()
                
                if all_signals:
                    st.success(f"🎯 Generated **{len(all_signals)}** trading signals!")
                    
                    # Convert signals to display format
                    signal_data = []
                    for signal in all_signals:
                        signal_data.append({
                            'Symbol': signal.symbol,
                            'Strategy': signal.setup,
                            'Entry': f"${signal.entry:.2f}",
                            'Stop': f"${signal.stop:.2f}",
                            'Target': f"${signal.target:.2f}",
                            'Size': f"{signal.size:,}",
                            'Risk': f"${abs(signal.entry - signal.stop) * signal.size:.0f}",
                            'Reward': f"${abs(signal.target - signal.entry) * signal.size:.0f}",
                            'R:R': f"{signal.risk_reward:.1f}R",
                            'Confidence': f"{signal.confidence:.0%}",
                            'Gap %': f"{signal.gap_percent:.1%}",
                            'RVOL': f"{signal.rvol:.1f}",
                            'Notes': signal.notes
                        })
                    
                    signals_df = pd.DataFrame(signal_data)
                    
                    # Display signals table
                    st.dataframe(
                        signals_df,
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=1,
                                format="%.0%%"
                            )
                        }
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_risk = sum(abs(s.entry - s.stop) * s.size for s in all_signals)
                    total_reward = sum(abs(s.target - s.entry) * s.size for s in all_signals)
                    avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
                    
                    col1.metric("Total Signals", len(all_signals))
                    col2.metric("Total Risk", f"${total_risk:,.0f}")
                    col3.metric("Potential Reward", f"${total_reward:,.0f}")
                    col4.metric("Avg Confidence", f"{avg_confidence:.0%}")
                    
                    # Download signals
                    signals_csv = pd.DataFrame([asdict(s) for s in all_signals]).to_csv(index=False)
                    st.download_button(
                        "📥 Download Trading Signals",
                        data=signals_csv,
                        file_name=f"trading_signals_{trade_date}.csv",
                        mime="text/csv"
                    )
                    
                    # Store signals for charts tab
                    st.session_state.trading_signals = all_signals
                    st.session_state.signal_symbols = selected_symbols
                    
                else:
                    st.info("No signals generated. Try different symbols or adjust your filters.")
            else:
                st.error("Please select symbols and strategies.")
    
    with tab3:
        st.subheader("📊 Trading Charts & Analysis")
        
        # Check if we have signals to display
        if 'trading_signals' in st.session_state and st.session_state.trading_signals:
            # Symbol selection for charting
            chart_symbols = list(set([s.symbol for s in st.session_state.trading_signals]))
            
            selected_chart_symbol = st.selectbox(
                "Select symbol to analyze:",
                chart_symbols,
                help="Choose a symbol that has generated signals"
            )
            
            if selected_chart_symbol:
                # Get the data and signals for this symbol
                df = get_stock_data(selected_chart_symbol, period="1d", interval="1m")
                symbol_signals = [s for s in st.session_state.trading_signals if s.symbol == selected_chart_symbol]
                
                if not df.empty:
                    # Create and display chart
                    fig = create_trading_chart(df, symbol_signals, selected_chart_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current market data
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    current_price = df.iloc[-1]['close']
                    day_open = df.iloc[0]['open']
                    day_high = df['high'].max()
                    day_low = df['low'].min()
                    day_volume = df['volume'].sum()
                    
                    day_change = current_price - day_open
                    day_change_pct = (day_change / day_open) * 100 if day_open > 0 else 0
                    
                    col1.metric("Current Price", f"${current_price:.2f}", f"{day_change:+.2f} ({day_change_pct:+.1f}%)")
                    col2.metric("Day High", f"${day_high:.2f}")
                    col3.metric("Day Low", f"${day_low:.2f}")
                    col4.metric("Day Volume", f"{day_volume:,.0f}")
                    col5.metric("Range", f"${day_high - day_low:.2f}")
                    
                    # Signal details for this symbol
                    if symbol_signals:
                        st.subheader(f"📈 Active Signals for {selected_chart_symbol}")
                        
                        for signal in symbol_signals:
                            with st.expander(f"{signal.setup} - {signal.confidence:.0%} Confidence"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Entry Details:**")
                                    st.write(f"Entry Price: ${signal.entry:.2f}")
                                    st.write(f"Stop Loss: ${signal.stop:.2f}")
                                    st.write(f"Target: ${signal.target:.2f}")
                                
                                with col2:
                                    st.write("**Position Details:**")
                                    st.write(f"Position Size: {signal.size:,} shares")
                                    st.write(f"Risk per Share: ${abs(signal.entry - signal.stop):.2f}")
                                    st.write(f"Total Risk: ${abs(signal.entry - signal.stop) * signal.size:.0f}")
                                
                                with col3:
                                    st.write("**Signal Quality:**")
                                    st.write(f"Confidence: {signal.confidence:.0%}")
                                    st.write(f"Gap %: {signal.gap_percent:.1%}")
                                    st.write(f"RVOL: {signal.rvol:.1f}x")
                                
                                st.write(f"**Notes:** {signal.notes}")
                else:
                    st.error(f"Could not load chart data for {selected_chart_symbol}")
        else:
            st.info("Generate trading signals first to see charts and analysis.")
            
            # Option to view any symbol without signals
            st.subheader("📈 View Any Symbol")
            manual_symbol = st.text_input("Enter symbol for chart analysis:", "AAPL")
            
            if st.button("Load Chart") and manual_symbol:
                df = get_stock_data(manual_symbol.upper(), period="1d", interval="1m")
                if not df.empty:
                    fig = create_trading_chart(df, [], manual_symbol.upper())
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Risk Management Tools")
        
        # Position size calculator
        st.write("### 🧮 Position Size Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            calc_entry = st.number_input("Entry Price ($)", value=10.00, min_value=0.01, step=0.01)
            calc_stop = st.number_input("Stop Loss ($)", value=9.50, min_value=0.01, step=0.01)
            calc_account = st.number_input("Account Size ($)", value=cfg['account_size'], min_value=1000, step=1000)
            calc_risk = st.slider("Risk per Trade (%)", 0.1, 5.0, cfg['risk_per_trade'] * 100) / 100
        
        with col2:
            if calc_entry > 0 and calc_stop > 0 and calc_entry != calc_stop:
                calc_size = position_size(calc_entry, calc_stop, calc_account, calc_risk)
                risk_per_share = abs(calc_entry - calc_stop)
                total_risk = risk_per_share * calc_size
                total_position_value = calc_entry * calc_size
                
                st.metric("Recommended Size", f"{calc_size:,} shares")
                st.metric("Risk per Share", f"${risk_per_share:.2f}")
                st.metric("Total Risk", f"${total_risk:.0f}")
                st.metric("Position Value", f"${total_position_value:,.0f}")
                
                if total_position_value > calc_account:
                    st.warning("⚠️ Position value exceeds account size!")
            else:
                st.info("Enter valid entry and stop prices to calculate position size.")
        
        # Daily P&L tracker
        st.write("### 📊 Daily P&L Tracker")
        
        # Initialize session state for P&L tracking
        if 'daily_trades' not in st.session_state:
            st.session_state.daily_trades = []
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_pnl = st.number_input("Trade P&L ($)", value=0.0, step=10.0)
            trade_symbol = st.text_input("Symbol (optional)", value="")
        
        with col2:
            if st.button("➕ Add Trade"):
                st.session_state.daily_trades.append({
                    'symbol': trade_symbol or 'N/A',
                    'pnl': trade_pnl,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                st.success(f"Added trade: ${trade_pnl:.2f}")
        
        with col3:
            if st.button("🗑️ Reset Day"):
                st.session_state.daily_trades = []
                st.success("Daily P&L reset")
        
        # Display daily stats
        if st.session_state.daily_trades:
            trades_df = pd.DataFrame(st.session_state.daily_trades)
            
            total_pnl = trades_df['pnl'].sum()
            trade_count = len(trades_df)
            winners = len(trades_df[trades_df['pnl'] > 0])
            losers = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winners / trade_count) * 100 if trade_count > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Daily P&L", f"${total_pnl:.2f}", f"{(total_pnl/cfg['account_size'])*100:.2f}%")
            col2.metric("Total Trades", trade_count)
            col3.metric("Winners", winners)
            col4.metric("Losers", losers)
            col5.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Check daily loss limit
            daily_loss_limit = cfg['account_size'] * cfg.get('max_daily_loss', 0.02)
            if total_pnl < -daily_loss_limit:
                st.error(f"🚨 **DAILY LOSS LIMIT EXCEEDED!** Stop trading. Loss: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            elif total_pnl < -daily_loss_limit * 0.75:
                st.warning(f"⚠️ **Approaching daily loss limit.** Current: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            
            # Show recent trades
            st.write("**Recent Trades:**")
            st.dataframe(trades_df.tail(10), use_container_width=True)
    
    with tab5:
        st.subheader("ℹ️ Trading Information & Disclaimers")
        
        # Strategy explanations
        st.write("### 🎯 Implemented Trading Strategies")
        
        with st.expander("1-Minute Opening Range Breakout (ORB)"):
            st.write("""
            **Strategy:** Break of the first 1-minute candle of regular trading hours.
            
            **Entry:** $0.01 above the high of the first 1-minute candle (9:30-9:31 AM ET)
            **Stop:** Below the low of the first candle minus buffer
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Strong gap-ups with good volume and range in the opening candle.
            """)
        
        with st.expander("Premarket High Break"):
            st.write("""
            **Strategy:** Breakout above the highest price reached during premarket trading.
            
            **Entry:** $0.01 above premarket high
            **Stop:** Below recent structural low minus buffer  
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Stocks with significant premarket activity and volume.
            """)
        
        with st.expander("Gap and Go"):
            st.write("""
            **Strategy:** Momentum continuation after significant overnight gap.
            
            **Entry:** Above current resistance levels
            **Stop:** Percentage-based or technical level
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Clean gaps with strong volume and clear direction.
            """)
        
        # Risk management info
        st.write("### ⚠️ Risk Management Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Position Sizing:**
            - Based on account risk percentage
            - Calculated from entry to stop distance
            - Never exceed account buying power
            - Adjust for volatility and liquidity
            """)
        
        with col2:
            st.write("""
            **Daily Limits:**
            - Maximum risk per trade: 0.5-1% of account
            - Daily loss limit: 2% of account  
            - Stop trading when limit reached
            - Review and adjust rules regularly
            """)
        
        # Important disclaimers
        st.write("### 📋 Important Disclaimers")
        
        st.error("""
        **⚠️ RISK WARNING:**
        - Trading involves substantial risk and is not suitable for all investors
        - Past performance does not guarantee future results  
        - You can lose more than your initial investment
        - Only trade with money you can afford to lose completely
        """)
        
        st.warning("""
        **📚 EDUCATIONAL USE ONLY:**
        - This application is for educational and research purposes
        - Not intended as investment or trading advice
        - Consult qualified financial professionals before trading
        - Practice with paper trading before risking real money
        """)
        
        st.info("""
        **🔧 TECHNICAL NOTES:**
        - Uses Yahoo Finance for free market data (15-20 minute delay)
        - Real-time data requires premium data feeds
        - Signals are based on historical patterns and technical analysis
        - Market conditions can change rapidly - always monitor positions
        """)
        
        # App information
        st.write("### 🚀 About This Application")
        st.write("""
        **Version:** 1.0 - Working Release
        **Data Source:** Yahoo Finance (Free)
        **Update Frequency:** Real-time for display, cached for performance
        **Deployment:** Streamlit Cloud
        
        **Features:**
        - Real-time market scanning
        - Multiple momentum trading strategies
        - Risk-based position sizing
        - Interactive charting and analysis  
        - Daily P&L tracking
        - Professional signal generation
        
        **Recommended Usage:**
        1. Run morning scan before market open
        2. Review highest-scoring opportunities
        3. Generate signals for selected symbols
        4. Analyze charts and confirm setups
        5. Calculate position sizes and risk
        6. Execute trades manually in your broker
        7. Track performance and adjust strategy
        
        ---
        **Remember:** The best traders prioritize risk management over profits. Trade smart, trade safe! 📈
        """)

if __name__ == "__main__":
    main(), '') for s in default_universe.split(',') if s.strip()]
    
    # Show symbol source
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Monitoring **{len(symbols)}** symbols")
    with col2:
        if st.button("🔄 Load Popular Stocks"):
            st.session_state.suggested_symbols = ",".join(popular_stocks[:15])
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Market Scanner",
        "📈 Signal Generation", 
        "📊 Trading Charts",
        "⚠️ Risk Management",
        "ℹ️ Information"
    ])
    
    with tab1:
        st.subheader("🔍 Market Opportunity Scanner")
        
        # Two scanner options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Option 1: Custom Symbol List**")
            if st.button("🚀 Run Custom Scan", type="primary"):
                if symbols:
                    with st.spinner("Scanning your custom symbols..."):
                        scan_results = scan_for_opportunities(symbols, cfg)
                    
                    if not scan_results.empty:
                        st.success(f"🎯 Found **{len(scan_results)}** opportunities!")
                        
                        # Display results
                        st.dataframe(
                            scan_results,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Opportunity Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        # Download option
                        csv = scan_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Custom Scan",
                            data=csv,
                            file_name=f"custom_scan_{trade_date}.csv",
                            mime="text/csv"
                        )
                        
                        # Store results for other tabs
                        st.session_state.scan_results = scan_results
                        
                    else:
                        st.warning("No opportunities found in your symbol list.")
                else:
                    st.error("Please enter symbols to scan.")
        
        with col2:
            st.write("**Option 2: Full Market Scan (Requires Alpaca API)**")
            alpaca_client = get_alpaca_client()
            
            if alpaca_client and ALPACA_AVAILABLE:
                if st.button("🌍 Scan Entire Market", type="secondary"):
                    with st.spinner("Scanning entire market... may take 30-60 seconds"):
                        df_scan = full_market_gap_scan(trade_date, cfg, alpaca_client)
                    
                    if not df_scan.empty:
                        st.success(f"🎯 Found **{len(df_scan)}** market opportunities!")
                        
                        # Display top 100 results
                        display_df = df_scan.head(100)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Score",
                                    min_value=0,
                                    max_value=100,
                                    format="%d"
                                )
                            }
                        )
                        
                        st.download_button(
                            "📥 Download Full Market Scan", 
                            df_scan.to_csv(index=False), 
                            f"full_market_scan_{trade_date}.csv"
                        )
                        
                        st.caption("💡 Tip: After scanning, copy symbols from results to use in the Signals tab.")
                        
                        # Store full market results
                        st.session_state.market_scan_results = df_scan
                        
                        # Quick copy feature for top symbols
                        top_symbols = df_scan.head(20)['Symbol'].tolist()
                        top_symbols_str = ",".join(top_symbols)
                        
                        st.text_area(
                            "🔥 Top 20 Symbols (copy for Signal Generation):",
                            value=top_symbols_str,
                            height=80,
                            help="Copy these symbols and paste into the Signal Generation tab"
                        )
                        
                    else:
                        st.warning("No opportunities found in full market scan. Try adjusting your filters.")
            else:
                st.info("""
                **Full market scanning requires Alpaca API credentials.**
                
                To enable:
                1. Get free Alpaca account at alpaca.markets
                2. Add your API keys to Streamlit secrets
                3. Refresh this app
                
                **Benefits of full market scan:**
                - Scans 8,000+ NYSE/NASDAQ/AMEX stocks automatically
                - Finds opportunities you might miss manually
                - Ranks all opportunities by momentum potential
                - Updates with real-time market data
                """)
                
                # Show how to add API keys
                with st.expander("📝 How to Add Alpaca API Keys"):
                    st.code("""
                    # Add to your Streamlit app secrets:
                    ALPACA_API_KEY = "your_api_key_here"
                    ALPACA_API_SECRET = "your_secret_key_here"
                    """)
        
        # Results summary
        if 'scan_results' in st.session_state or 'market_scan_results' in st.session_state:
            st.divider()
            st.write("### 📊 Scanner Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'scan_results' in st.session_state:
                    custom_count = len(st.session_state.scan_results)
                    st.metric("Custom Scan Results", custom_count)
            
            with col2:
                if 'market_scan_results' in st.session_state:
                    market_count = len(st.session_state.market_scan_results)
                    st.metric("Full Market Results", market_count)
    
    with tab2:
        st.subheader("📈 Trading Signal Generation")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        with col1:
            strategies = st.multiselect(
                "Select Trading Strategies:",
                ["1m_ORB", "PM_High_Break", "Gap_and_Go"],
                default=["1m_ORB", "PM_High_Break"],
                help="Choose which momentum strategies to apply"
            )
        
        with col2:
            max_signals = st.slider("Max Signals per Symbol", 1, 5, 2)
        
        # Symbol selection for signals
        available_symbols = symbols[:10]  # Limit for performance
        selected_symbols = st.multiselect(
            "Select symbols for signal generation:",
            available_symbols,
            default=available_symbols[:5],
            help="Select up to 10 symbols for detailed signal analysis"
        )
        
        if st.button("🎯 Generate Trading Signals", type="primary"):
            if selected_symbols and strategies:
                all_signals = []
                
                progress_bar = st.progress(0)
                
                for i, symbol in enumerate(selected_symbols):
                    progress_bar.progress((i + 1) / len(selected_symbols))
                    
                    # Get intraday data
                    df = get_stock_data(symbol, period="1d", interval="1m")
                    
                    if df.empty:
                        continue
                    
                    # Apply selected strategies
                    if "1m_ORB" in strategies:
                        orb_signals = one_minute_orb_strategy(df, symbol, cfg)
                        all_signals.extend(orb_signals[:max_signals])
                    
                    if "PM_High_Break" in strategies:
                        pm_signals = premarket_high_break_strategy(df, symbol, cfg)
                        all_signals.extend(pm_signals[:max_signals])
                    
                    if "Gap_and_Go" in strategies:
                        gap_signals = gap_and_go_strategy(df, symbol, cfg)
                        all_signals.extend(gap_signals[:max_signals])
                
                progress_bar.empty()
                
                if all_signals:
                    st.success(f"🎯 Generated **{len(all_signals)}** trading signals!")
                    
                    # Convert signals to display format
                    signal_data = []
                    for signal in all_signals:
                        signal_data.append({
                            'Symbol': signal.symbol,
                            'Strategy': signal.setup,
                            'Entry': f"${signal.entry:.2f}",
                            'Stop': f"${signal.stop:.2f}",
                            'Target': f"${signal.target:.2f}",
                            'Size': f"{signal.size:,}",
                            'Risk': f"${abs(signal.entry - signal.stop) * signal.size:.0f}",
                            'Reward': f"${abs(signal.target - signal.entry) * signal.size:.0f}",
                            'R:R': f"{signal.risk_reward:.1f}R",
                            'Confidence': f"{signal.confidence:.0%}",
                            'Gap %': f"{signal.gap_percent:.1%}",
                            'RVOL': f"{signal.rvol:.1f}",
                            'Notes': signal.notes
                        })
                    
                    signals_df = pd.DataFrame(signal_data)
                    
                    # Display signals table
                    st.dataframe(
                        signals_df,
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=1,
                                format="%.0%%"
                            )
                        }
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_risk = sum(abs(s.entry - s.stop) * s.size for s in all_signals)
                    total_reward = sum(abs(s.target - s.entry) * s.size for s in all_signals)
                    avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
                    
                    col1.metric("Total Signals", len(all_signals))
                    col2.metric("Total Risk", f"${total_risk:,.0f}")
                    col3.metric("Potential Reward", f"${total_reward:,.0f}")
                    col4.metric("Avg Confidence", f"{avg_confidence:.0%}")
                    
                    # Download signals
                    signals_csv = pd.DataFrame([asdict(s) for s in all_signals]).to_csv(index=False)
                    st.download_button(
                        "📥 Download Trading Signals",
                        data=signals_csv,
                        file_name=f"trading_signals_{trade_date}.csv",
                        mime="text/csv"
                    )
                    
                    # Store signals for charts tab
                    st.session_state.trading_signals = all_signals
                    st.session_state.signal_symbols = selected_symbols
                    
                else:
                    st.info("No signals generated. Try different symbols or adjust your filters.")
            else:
                st.error("Please select symbols and strategies.")
    
    with tab3:
        st.subheader("📊 Trading Charts & Analysis")
        
        # Check if we have signals to display
        if 'trading_signals' in st.session_state and st.session_state.trading_signals:
            # Symbol selection for charting
            chart_symbols = list(set([s.symbol for s in st.session_state.trading_signals]))
            
            selected_chart_symbol = st.selectbox(
                "Select symbol to analyze:",
                chart_symbols,
                help="Choose a symbol that has generated signals"
            )
            
            if selected_chart_symbol:
                # Get the data and signals for this symbol
                df = get_stock_data(selected_chart_symbol, period="1d", interval="1m")
                symbol_signals = [s for s in st.session_state.trading_signals if s.symbol == selected_chart_symbol]
                
                if not df.empty:
                    # Create and display chart
                    fig = create_trading_chart(df, symbol_signals, selected_chart_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current market data
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    current_price = df.iloc[-1]['close']
                    day_open = df.iloc[0]['open']
                    day_high = df['high'].max()
                    day_low = df['low'].min()
                    day_volume = df['volume'].sum()
                    
                    day_change = current_price - day_open
                    day_change_pct = (day_change / day_open) * 100 if day_open > 0 else 0
                    
                    col1.metric("Current Price", f"${current_price:.2f}", f"{day_change:+.2f} ({day_change_pct:+.1f}%)")
                    col2.metric("Day High", f"${day_high:.2f}")
                    col3.metric("Day Low", f"${day_low:.2f}")
                    col4.metric("Day Volume", f"{day_volume:,.0f}")
                    col5.metric("Range", f"${day_high - day_low:.2f}")
                    
                    # Signal details for this symbol
                    if symbol_signals:
                        st.subheader(f"📈 Active Signals for {selected_chart_symbol}")
                        
                        for signal in symbol_signals:
                            with st.expander(f"{signal.setup} - {signal.confidence:.0%} Confidence"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Entry Details:**")
                                    st.write(f"Entry Price: ${signal.entry:.2f}")
                                    st.write(f"Stop Loss: ${signal.stop:.2f}")
                                    st.write(f"Target: ${signal.target:.2f}")
                                
                                with col2:
                                    st.write("**Position Details:**")
                                    st.write(f"Position Size: {signal.size:,} shares")
                                    st.write(f"Risk per Share: ${abs(signal.entry - signal.stop):.2f}")
                                    st.write(f"Total Risk: ${abs(signal.entry - signal.stop) * signal.size:.0f}")
                                
                                with col3:
                                    st.write("**Signal Quality:**")
                                    st.write(f"Confidence: {signal.confidence:.0%}")
                                    st.write(f"Gap %: {signal.gap_percent:.1%}")
                                    st.write(f"RVOL: {signal.rvol:.1f}x")
                                
                                st.write(f"**Notes:** {signal.notes}")
                else:
                    st.error(f"Could not load chart data for {selected_chart_symbol}")
        else:
            st.info("Generate trading signals first to see charts and analysis.")
            
            # Option to view any symbol without signals
            st.subheader("📈 View Any Symbol")
            manual_symbol = st.text_input("Enter symbol for chart analysis:", "AAPL")
            
            if st.button("Load Chart") and manual_symbol:
                df = get_stock_data(manual_symbol.upper(), period="1d", interval="1m")
                if not df.empty:
                    fig = create_trading_chart(df, [], manual_symbol.upper())
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Risk Management Tools")
        
        # Position size calculator
        st.write("### 🧮 Position Size Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            calc_entry = st.number_input("Entry Price ($)", value=10.00, min_value=0.01, step=0.01)
            calc_stop = st.number_input("Stop Loss ($)", value=9.50, min_value=0.01, step=0.01)
            calc_account = st.number_input("Account Size ($)", value=cfg['account_size'], min_value=1000, step=1000)
            calc_risk = st.slider("Risk per Trade (%)", 0.1, 5.0, cfg['risk_per_trade'] * 100) / 100
        
        with col2:
            if calc_entry > 0 and calc_stop > 0 and calc_entry != calc_stop:
                calc_size = position_size(calc_entry, calc_stop, calc_account, calc_risk)
                risk_per_share = abs(calc_entry - calc_stop)
                total_risk = risk_per_share * calc_size
                total_position_value = calc_entry * calc_size
                
                st.metric("Recommended Size", f"{calc_size:,} shares")
                st.metric("Risk per Share", f"${risk_per_share:.2f}")
                st.metric("Total Risk", f"${total_risk:.0f}")
                st.metric("Position Value", f"${total_position_value:,.0f}")
                
                if total_position_value > calc_account:
                    st.warning("⚠️ Position value exceeds account size!")
            else:
                st.info("Enter valid entry and stop prices to calculate position size.")
        
        # Daily P&L tracker
        st.write("### 📊 Daily P&L Tracker")
        
        # Initialize session state for P&L tracking
        if 'daily_trades' not in st.session_state:
            st.session_state.daily_trades = []
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_pnl = st.number_input("Trade P&L ($)", value=0.0, step=10.0)
            trade_symbol = st.text_input("Symbol (optional)", value="")
        
        with col2:
            if st.button("➕ Add Trade"):
                st.session_state.daily_trades.append({
                    'symbol': trade_symbol or 'N/A',
                    'pnl': trade_pnl,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                st.success(f"Added trade: ${trade_pnl:.2f}")
        
        with col3:
            if st.button("🗑️ Reset Day"):
                st.session_state.daily_trades = []
                st.success("Daily P&L reset")
        
        # Display daily stats
        if st.session_state.daily_trades:
            trades_df = pd.DataFrame(st.session_state.daily_trades)
            
            total_pnl = trades_df['pnl'].sum()
            trade_count = len(trades_df)
            winners = len(trades_df[trades_df['pnl'] > 0])
            losers = len(trades_df[trades_df['pnl'] < 0])
            win_rate = (winners / trade_count) * 100 if trade_count > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Daily P&L", f"${total_pnl:.2f}", f"{(total_pnl/cfg['account_size'])*100:.2f}%")
            col2.metric("Total Trades", trade_count)
            col3.metric("Winners", winners)
            col4.metric("Losers", losers)
            col5.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Check daily loss limit
            daily_loss_limit = cfg['account_size'] * cfg.get('max_daily_loss', 0.02)
            if total_pnl < -daily_loss_limit:
                st.error(f"🚨 **DAILY LOSS LIMIT EXCEEDED!** Stop trading. Loss: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            elif total_pnl < -daily_loss_limit * 0.75:
                st.warning(f"⚠️ **Approaching daily loss limit.** Current: ${total_pnl:.2f} (Limit: ${-daily_loss_limit:.2f})")
            
            # Show recent trades
            st.write("**Recent Trades:**")
            st.dataframe(trades_df.tail(10), use_container_width=True)
    
    with tab5:
        st.subheader("ℹ️ Trading Information & Disclaimers")
        
        # Strategy explanations
        st.write("### 🎯 Implemented Trading Strategies")
        
        with st.expander("1-Minute Opening Range Breakout (ORB)"):
            st.write("""
            **Strategy:** Break of the first 1-minute candle of regular trading hours.
            
            **Entry:** $0.01 above the high of the first 1-minute candle (9:30-9:31 AM ET)
            **Stop:** Below the low of the first candle minus buffer
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Strong gap-ups with good volume and range in the opening candle.
            """)
        
        with st.expander("Premarket High Break"):
            st.write("""
            **Strategy:** Breakout above the highest price reached during premarket trading.
            
            **Entry:** $0.01 above premarket high
            **Stop:** Below recent structural low minus buffer  
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Stocks with significant premarket activity and volume.
            """)
        
        with st.expander("Gap and Go"):
            st.write("""
            **Strategy:** Momentum continuation after significant overnight gap.
            
            **Entry:** Above current resistance levels
            **Stop:** Percentage-based or technical level
            **Target:** Risk-reward multiple (typically 2R)
            
            **Best for:** Clean gaps with strong volume and clear direction.
            """)
        
        # Risk management info
        st.write("### ⚠️ Risk Management Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Position Sizing:**
            - Based on account risk percentage
            - Calculated from entry to stop distance
            - Never exceed account buying power
            - Adjust for volatility and liquidity
            """)
        
        with col2:
            st.write("""
            **Daily Limits:**
            - Maximum risk per trade: 0.5-1% of account
            - Daily loss limit: 2% of account  
            - Stop trading when limit reached
            - Review and adjust rules regularly
            """)
        
        # Important disclaimers
        st.write("### 📋 Important Disclaimers")
        
        st.error("""
        **⚠️ RISK WARNING:**
        - Trading involves substantial risk and is not suitable for all investors
        - Past performance does not guarantee future results  
        - You can lose more than your initial investment
        - Only trade with money you can afford to lose completely
        """)
        
        st.warning("""
        **📚 EDUCATIONAL USE ONLY:**
        - This application is for educational and research purposes
        - Not intended as investment or trading advice
        - Consult qualified financial professionals before trading
        - Practice with paper trading before risking real money
        """)
        
        st.info("""
        **🔧 TECHNICAL NOTES:**
        - Uses Yahoo Finance for free market data (15-20 minute delay)
        - Real-time data requires premium data feeds
        - Signals are based on historical patterns and technical analysis
        - Market conditions can change rapidly - always monitor positions
        """)
        
        # App information
        st.write("### 🚀 About This Application")
        st.write("""
        **Version:** 1.0 - Working Release
        **Data Source:** Yahoo Finance (Free)
        **Update Frequency:** Real-time for display, cached for performance
        **Deployment:** Streamlit Cloud
        
        **Features:**
        - Real-time market scanning
        - Multiple momentum trading strategies
        - Risk-based position sizing
        - Interactive charting and analysis  
        - Daily P&L tracking
        - Professional signal generation
        
        **Recommended Usage:**
        1. Run morning scan before market open
        2. Review highest-scoring opportunities
        3. Generate signals for selected symbols
        4. Analyze charts and confirm setups
        5. Calculate position sizes and risk
        6. Execute trades manually in your broker
        7. Track performance and adjust strategy
        
        ---
        **Remember:** The best traders prioritize risk management over profits. Trade smart, trade safe! 📈
        """)

if __name__ == "__main__":
    main()
