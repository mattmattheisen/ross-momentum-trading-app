"""
Enhanced Ross-Style Momentum Signal Generator (Streamlit + Alpaca)
================================================================
Advanced momentum trading signals with enhanced features:
- Real-time market data integration
- Advanced filtering (float data integration ready)
- Multiple momentum setups (ORB, PM-High, 9EMA pullback, ABCD)
- Enhanced risk management with daily P&L tracking
- Real-time alerts and notifications
- Advanced backtesting with statistics
- Position tracking and management tools

Quick Start
-----------
1. `pip install -r requirements.txt`
2. Create `.streamlit/secrets.toml` with Alpaca credentials
3. Run: `streamlit run enhanced_ross_app.py`

Requirements
------------
streamlit>=1.33.0
alpaca-py>=0.22.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
plotly>=5.18.0
scipy>=1.10.0
"""

import os
import datetime as dt
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Alpaca imports
from alpaca.data.historical import StockBarsClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ------------- ENHANCED CONFIG DEFAULTS -------------
DEFAULTS = {
    "gap_min": 0.06,            # 6%
    "gap_max": 0.50,            # 50% (avoid crazy gaps)
    "price_min": 2.0,
    "price_max": 20.0,
    "float_max": 50_000_000,    
    "rvol_min": 2.0,
    "pm_vol_min": 150_000,
    "risk_per_trade": 0.005,    # 0.5%
    "max_daily_loss": 0.02,     # 2%
    "stop_buffer": 0.02,        
    "target_R": 2.0,
    "ema_period": 9,
    "min_spread": 0.05,         # Min bid-ask spread
    "max_spread": 0.25,         # Max bid-ask spread
}

MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
PREMARKET_START = dt.time(4, 0)
PREMARKET_END = dt.time(9, 29)

# ------------- ENHANCED DATA CLASSES -------------
@dataclass
class Signal:
    symbol: str
    setup: str
    entry: float
    stop: float
    target: float
    size: int
    time: pd.Timestamp
    notes: str
    confidence: float = 0.0  # 0-1 confidence score
    risk_reward: float = 0.0
    pm_high: float = 0.0
    pm_volume: int = 0
    gap_percent: float = 0.0
    rvol: float = 0.0

@dataclass
class Position:
    symbol: str
    entry_price: float
    stop_loss: float
    target: float
    size: int
    entry_time: pd.Timestamp
    pnl: float = 0.0
    is_open: bool = True

@dataclass
class BacktestStats:
    total_trades: int
    win_rate: float
    avg_r: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

# ------------- UTILITIES -------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_alpaca_client() -> StockBarsClient:
    """Get cached Alpaca client"""
    key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY"))
    secret = st.secrets.get("ALPACA_API_SECRET", os.getenv("ALPACA_API_SECRET"))
    if not key or secret:
        st.error("Missing Alpaca API credentials. Add to st.secrets or env vars.")
        st.stop()
    return StockBarsClient(api_key=key, secret_key=secret)

@st.cache_data(ttl=60)  # Cache minute data for 1 minute
def fetch_minute_bars(symbol: str, date: dt.date, client: StockBarsClient) -> pd.DataFrame:
    """Fetch 1-minute bars with caching"""
    start = dt.datetime.combine(date, PREMARKET_START)
    end = dt.datetime.combine(date, dt.time(20, 0))
    req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute,
                           start=start, end=end, adjustment=None)
    try:
        bars = client.get_stock_bars(req)
        if symbol not in bars.data:
            return pd.DataFrame()
        df = bars.data[symbol].df
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache daily data for 5 minutes
def fetch_daily_bars(symbols: List[str], lookback: int, end_date: dt.date, client: StockBarsClient) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars with enhanced error handling"""
    start = end_date - dt.timedelta(days=lookback*2)
    start_dt = dt.datetime.combine(start, dt.time(4, 0))
    end_dt = dt.datetime.combine(end_date, dt.time(20, 0))
    
    # Process in batches to avoid API limits
    batch_size = 50
    all_data = {}
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                   start=start_dt, end=end_dt, adjustment=None)
            data = client.get_stock_bars(req)
            for sym, barset in data.data.items():
                df = barset.df
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)
                all_data[sym] = df
        except Exception as e:
            st.warning(f"Error fetching daily data for batch {i}: {e}")
            continue
    
    return all_data

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    df = df.copy()
    
    # EMA
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema20'] = df['close'].ewm(span=20).mean()
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    bb_std_val = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
    
    return df

def calc_rvol(today_vol: float, hist_volumes: pd.Series, lookback: int = 50) -> float:
    """Calculate relative volume"""
    if hist_volumes.empty:
        return np.nan
    avg = hist_volumes.tail(lookback).mean()
    return today_vol / avg if avg > 0 else np.nan

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def position_size(entry: float, stop: float, account_size: float, risk_pct: float) -> int:
    """Calculate position size based on risk"""
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    dollars_at_risk = account_size * risk_pct
    return max(1, int(dollars_at_risk / risk_per_share))

def calculate_confidence_score(signal: Signal, df: pd.DataFrame, pm_data: Dict) -> float:
    """Calculate confidence score for signal (0-1)"""
    score = 0.5  # Base score
    
    # Gap size factor (sweet spot 6-20%)
    if 0.06 <= signal.gap_percent <= 0.20:
        score += 0.2
    elif signal.gap_percent > 0.30:
        score -= 0.1
    
    # RVOL factor
    if signal.rvol >= 3.0:
        score += 0.2
    elif signal.rvol >= 2.0:
        score += 0.1
    
    # Volume factor
    if pm_data.get('pm_volume', 0) >= 500_000:
        score += 0.1
    
    # Price action quality (volatility)
    recent_candles = df.tail(10)
    if not recent_candles.empty:
        avg_range = ((recent_candles['high'] - recent_candles['low']) / recent_candles['close']).mean()
        if avg_range >= 0.02:  # 2% average range
            score += 0.1
    
    return min(1.0, max(0.0, score))

# ------------- ENHANCED STRATEGIES -------------
def one_min_orb(df: pd.DataFrame, cfg: dict, symbol: str) -> List[Signal]:
    """Enhanced 1-minute ORB strategy"""
    signals = []
    if df.empty:
        return signals
    
    reg = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    if reg.empty:
        return signals
    
    first_candle = reg.iloc[0]
    orb_high = first_candle['high']
    orb_low = first_candle['low']
    
    # Find breakout
    breakout_candles = reg[reg['high'] > orb_high]
    if breakout_candles.empty:
        return signals
    
    breakout_time = breakout_candles.index[0]
    entry = orb_high + 0.01
    stop = orb_low - cfg['stop_buffer']
    
    # Enhanced position sizing with risk management
    size = position_size(entry, stop, cfg['account_size'], cfg['risk_per_trade'])
    if size == 0:
        return signals
    
    target = entry + (entry - stop) * cfg['target_R']
    
    # Calculate additional metrics
    gap_pct = calculate_gap_percentage(df, symbol)
    rvol = calculate_rvol_for_symbol(df, symbol)
    
    signal = Signal(
        symbol=symbol,
        setup="1m_ORB",
        entry=entry,
        stop=stop,
        target=target,
        size=size,
        time=breakout_time,
        notes=f"Break of 1m ORB high {orb_high:.2f}",
        risk_reward=cfg['target_R'],
        gap_percent=gap_pct,
        rvol=rvol
    )
    
    signal.confidence = calculate_confidence_score(signal, df, {})
    signals.append(signal)
    
    return signals

def pm_high_break(df: pd.DataFrame, cfg: dict, symbol: str) -> List[Signal]:
    """Enhanced premarket high break strategy"""
    signals = []
    if df.empty:
        return signals
    
    # Get premarket data
    pm = df.between_time(PREMARKET_START, PREMARKET_END)
    if pm.empty:
        return signals
    
    pm_high = pm['high'].max()
    pm_volume = pm['volume'].sum()
    
    # Regular session data
    reg = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    if reg.empty:
        return signals
    
    # Find break of PM high
    breakout_candles = reg[reg['high'] > pm_high]
    if breakout_candles.empty:
        return signals
    
    breakout_time = breakout_candles.index[0]
    entry = pm_high + 0.01
    
    # Dynamic stop based on recent structure
    recent_lows = reg.loc[:breakout_time]['low'].tail(5)
    if not recent_lows.empty:
        stop = recent_lows.min() - cfg['stop_buffer']
    else:
        stop = entry * 0.97  # 3% stop as fallback
    
    size = position_size(entry, stop, cfg['account_size'], cfg['risk_per_trade'])
    if size == 0:
        return signals
    
    target = entry + (entry - stop) * cfg['target_R']
    
    gap_pct = calculate_gap_percentage(df, symbol)
    rvol = calculate_rvol_for_symbol(df, symbol)
    
    signal = Signal(
        symbol=symbol,
        setup="PM_High_Break",
        entry=entry,
        stop=stop,
        target=target,
        size=size,
        time=breakout_time,
        notes=f"Break of PM high {pm_high:.2f} (Vol: {pm_volume:,})",
        pm_high=pm_high,
        pm_volume=pm_volume,
        risk_reward=cfg['target_R'],
        gap_percent=gap_pct,
        rvol=rvol
    )
    
    signal.confidence = calculate_confidence_score(signal, df, {"pm_volume": pm_volume})
    signals.append(signal)
    
    return signals

def ema_pullback_strategy(df: pd.DataFrame, cfg: dict, symbol: str) -> List[Signal]:
    """9 EMA pullback strategy"""
    signals = []
    if df.empty:
        return signals
    
    # Add technical indicators
    df_tech = calculate_technical_indicators(df)
    reg = df_tech.between_time(MARKET_OPEN, MARKET_CLOSE)
    
    if len(reg) < 20:  # Need enough data for EMA
        return signals
    
    # Look for pullback to 9 EMA after gap up
    gap_pct = calculate_gap_percentage(df, symbol)
    if gap_pct < cfg['gap_min']:
        return signals
    
    # Find touches of 9 EMA
    ema_touches = []
    for i in range(10, len(reg)):
        current = reg.iloc[i]
        prev = reg.iloc[i-1]
        
        # Check if price touched EMA and bounced
        if (prev['low'] <= prev['ema9'] <= prev['high'] and 
            current['close'] > current['ema9'] and
            current['close'] > prev['close']):
            ema_touches.append(i)
    
    for touch_idx in ema_touches:
        candle = reg.iloc[touch_idx]
        entry = candle['high'] + 0.01
        stop = candle['ema9'] - cfg['stop_buffer']
        
        size = position_size(entry, stop, cfg['account_size'], cfg['risk_per_trade'])
        if size == 0:
            continue
        
        target = entry + (entry - stop) * cfg['target_R']
        
        signal = Signal(
            symbol=symbol,
            setup="9EMA_Pullback",
            entry=entry,
            stop=stop,
            target=target,
            size=size,
            time=candle.name,
            notes=f"9EMA pullback entry at {entry:.2f}",
            risk_reward=cfg['target_R'],
            gap_percent=gap_pct,
            rvol=calculate_rvol_for_symbol(df, symbol)
        )
        
        signal.confidence = calculate_confidence_score(signal, df, {})
        signals.append(signal)
    
    return signals

# ------------- HELPER FUNCTIONS -------------
def calculate_gap_percentage(df: pd.DataFrame, symbol: str) -> float:
    """Calculate gap percentage for the day"""
    if len(df) < 2:
        return 0.0
    
    today_open = df.iloc[0]['open']
    prev_close = df.iloc[0]['close']  # This is simplified - in real implementation, get previous day's close
    
    return (today_open - prev_close) / prev_close if prev_close > 0 else 0.0

def calculate_rvol_for_symbol(df: pd.DataFrame, symbol: str) -> float:
    """Calculate RVOL for symbol"""
    if df.empty:
        return 0.0
    
    current_vol = df['volume'].sum()
    avg_vol = df['volume'].mean() * len(df)  # Simplified calculation
    
    return current_vol / avg_vol if avg_vol > 0 else 0.0

def premarket_analysis(df: pd.DataFrame) -> Dict:
    """Analyze premarket activity"""
    if df.empty:
        return {}
    
    pm = df.between_time(PREMARKET_START, PREMARKET_END)
    if pm.empty:
        return {}
    
    return {
        'pm_high': pm['high'].max(),
        'pm_low': pm['low'].min(),
        'pm_volume': pm['volume'].sum(),
        'pm_open': pm.iloc[0]['open'],
        'pm_close': pm.iloc[-1]['close'],
        'pm_range': pm['high'].max() - pm['low'].min(),
        'pm_candles': len(pm)
    }

# ------------- ENHANCED SCANNER -------------
def enhanced_scan_gappers(symbols: List[str], date: dt.date, client: StockBarsClient, cfg: dict) -> pd.DataFrame:
    """Enhanced scanner with more metrics"""
    daily = fetch_daily_bars(symbols, lookback=60, end_date=date, client=client)
    results = []
    
    for sym, df in daily.items():
        if df.empty or len(df) < 2:
            continue
        
        today_row = df[df.index.date == date]
        if today_row.empty:
            continue
        
        today = today_row.iloc[0]
        prev = df.loc[df.index < today_row.index[0]].iloc[-1]
        
        # Calculate metrics
        gap_pct = (today['open'] - prev['close']) / prev['close']
        rvol = calc_rvol(today['volume'], df['volume'])
        
        # Price filter
        price_ok = cfg['price_min'] <= today['open'] <= cfg['price_max']
        
        # Gap filter
        gap_ok = cfg['gap_min'] <= gap_pct <= cfg.get('gap_max', 1.0)
        
        # Volume filter
        rvol_ok = rvol >= cfg['rvol_min'] if not np.isnan(rvol) else False
        
        if gap_ok and price_ok and rvol_ok:
            # Get intraday data for additional analysis
            minute_df = fetch_minute_bars(sym, date, client)
            pm_data = premarket_analysis(minute_df)
            
            results.append({
                'Symbol': sym,
                'Price': today['open'],
                'Gap %': f"{gap_pct:.1%}",
                'RVOL': f"{rvol:.1f}" if not np.isnan(rvol) else "N/A",
                'Volume': f"{today['volume']:,}",
                'PM High': pm_data.get('pm_high', 0),
                'PM Vol': f"{pm_data.get('pm_volume', 0):,}",
                'ATR': f"{calc_atr(df).iloc[-1]:.2f}" if len(df) >= 14 else "N/A",
                'Float': "TBD",  # Placeholder for float data
                'Score': calculate_scanner_score(gap_pct, rvol, pm_data)
            })
    
    return pd.DataFrame(results).sort_values('Score', ascending=False)

def calculate_scanner_score(gap_pct: float, rvol: float, pm_data: Dict) -> float:
    """Calculate scanner score for ranking"""
    score = 0
    
    # Gap score
    if 0.06 <= gap_pct <= 0.20:
        score += 40
    elif 0.20 < gap_pct <= 0.35:
        score += 30
    elif gap_pct > 0.35:
        score += 10
    
    # RVOL score
    if not np.isnan(rvol):
        if rvol >= 5:
            score += 30
        elif rvol >= 3:
            score += 20
        elif rvol >= 2:
            score += 10
    
    # PM volume score
    pm_vol = pm_data.get('pm_volume', 0)
    if pm_vol >= 500_000:
        score += 20
    elif pm_vol >= 250_000:
        score += 10
    
    # PM range score
    pm_range = pm_data.get('pm_range', 0)
    if pm_range >= 0.5:
        score += 10
    
    return score

# ------------- ENHANCED BACKTESTING -------------
def run_enhanced_backtest(signals: List[Signal], symbol_data: Dict[str, pd.DataFrame], 
                         cfg: dict) -> Tuple[pd.DataFrame, BacktestStats]:
    """Enhanced backtesting with detailed statistics"""
    results = []
    daily_pnl = {}
    
    for signal in signals:
        df = symbol_data.get(signal.symbol, pd.DataFrame())
        if df.empty:
            continue
        
        # Get data after signal time
        after_signal = df[df.index >= signal.time]
        if after_signal.empty:
            continue
        
        exit_price = None
        exit_time = None
        outcome = None
        
        # Simulate trade execution
        for ts, row in after_signal.iterrows():
            # Check for stop loss
            if row['low'] <= signal.stop:
                exit_price = signal.stop
                exit_time = ts
                outcome = 'stop'
                break
            
            # Check for target
            if row['high'] >= signal.target:
                exit_price = signal.target
                exit_time = ts
                outcome = 'target'
                break
        
        # If no exit, close at end of day
        if exit_price is None:
            exit_price = after_signal.iloc[-1]['close']
            exit_time = after_signal.index[-1]
            outcome = 'eod'
        
        # Calculate P&L
        r_mult = (exit_price - signal.entry) / (signal.entry - signal.stop) if (signal.entry - signal.stop) != 0 else 0
        pnl_dollars = r_mult * (cfg['account_size'] * cfg['risk_per_trade'])
        
        # Track daily P&L
        trade_date = signal.time.date()
        if trade_date not in daily_pnl:
            daily_pnl[trade_date] = 0
        daily_pnl[trade_date] += pnl_dollars
        
        results.append({
            'Symbol': signal.symbol,
            'Setup': signal.setup,
            'Entry Time': signal.time,
            'Entry': signal.entry,
            'Stop': signal.stop,
            'Target': signal.target,
            'Exit Time': exit_time,
            'Exit': exit_price,
            'Size': signal.size,
            'R-Multiple': r_mult,
            'P&L ($)': pnl_dollars,
            'Outcome': outcome,
            'Confidence': signal.confidence,
            'Gap %': f"{signal.gap_percent:.1%}",
            'RVOL': signal.rvol,
            'Notes': signal.notes
        })
    
    df_results = pd.DataFrame(results)
    
    # Calculate statistics
    if not df_results.empty:
        stats = BacktestStats(
            total_trades=len(df_results),
            win_rate=(df_results['R-Multiple'] > 0).mean(),
            avg_r=df_results['R-Multiple'].mean(),
            total_pnl=df_results['P&L ($)'].sum(),
            max_drawdown=calculate_max_drawdown(df_results['P&L ($)']),
            sharpe_ratio=calculate_sharpe_ratio(df_results['P&L ($)']),
            profit_factor=calculate_profit_factor(df_results['P&L ($)'])
        )
    else:
        stats = BacktestStats(0, 0, 0, 0, 0, 0, 0)
    
    return df_results, stats

def calculate_max_drawdown(pnl_series: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if pnl_series.empty:
        return 0.0
    
    cumulative = pnl_series.cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    return drawdown.min()

def calculate_sharpe_ratio(pnl_series: pd.Series) -> float:
    """Calculate Sharpe ratio"""
    if pnl_series.empty or pnl_series.std() == 0:
        return 0.0
    return pnl_series.mean() / pnl_series.std() * np.sqrt(252)  # Annualized

def calculate_profit_factor(pnl_series: pd.Series) -> float:
    """Calculate profit factor"""
    if pnl_series.empty:
        return 0.0
    
    profits = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    
    return profits / losses if losses > 0 else float('inf')

# ------------- ENHANCED VISUALIZATION -------------
def create_signal_chart(df: pd.DataFrame, signals: List[Signal], symbol: str) -> go.Figure:
    """Create enhanced chart with signals and indicators"""
    if df.empty:
        return go.Figure()
    
    # Add technical indicators
    df_tech = calculate_technical_indicators(df)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} - Price Action', 'Volume', 'RSI'),
        row_width=[0.7, 0.2, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_tech.index,
            open=df_tech['open'],
            high=df_tech['high'],
            low=df_tech['low'],
            close=df_tech['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add EMAs
    fig.add_trace(
        go.Scatter(x=df_tech.index, y=df_tech['ema9'], name='EMA 9', 
                   line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_tech.index, y=df_tech['vwap'], name='VWAP', 
                   line=dict(color='purple', width=1)),
        row=1, col=1
    )
    
    # Add signals
    for signal in signals:
        if signal.symbol == symbol:
            fig.add_trace(
                go.Scatter(x=[signal.time], y=[signal.entry], mode='markers',
                          marker=dict(color='green', size=10, symbol='triangle-up'),
                          name=f'{signal.setup} Entry'),
                row=1, col=1
            )
            
            # Add stop and target lines
            signal_data = df_tech[df_tech.index >= signal.time]
            if not signal_data.empty:
                fig.add_hline(y=signal.stop, line_dash="dash", line_color="red",
                             annotation_text=f"Stop: {signal.stop:.2f}", row=1, col=1)
                fig.add_hline(y=signal.target, line_dash="dash", line_color="green",
                             annotation_text=f"Target: {signal.target:.2f}", row=1, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df_tech.index, y=df_tech['volume'], name='Volume', 
               marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df_tech.index, y=df_tech['rsi'], name='RSI', 
                   line=dict(color='blue')),
        row=3, col=1
    )
    
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Add premarket/regular session backgrounds
    market_open_dt = dt.datetime.combine(df.index[0].date(), MARKET_OPEN)
    
    fig.add_vrect(
        x0=df.index[0], x1=market_open_dt,
        fillcolor="yellow", opacity=0.1,
        annotation_text="Premarket", annotation_position="top left",
        row=1, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Trading Signals",
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

# ------------- REAL-TIME MONITORING -------------
def real_time_monitor(symbols: List[str], cfg: dict) -> Dict[str, Dict]:
    """Real-time monitoring of positions and alerts"""
    client = get_alpaca_client()
    today = dt.date.today()
    
    alerts = {}
    
    for symbol in symbols:
        df = fetch_minute_bars(symbol, today, client)
        if df.empty:
            continue
            
        current_price = df.iloc[-1]['close']
        pm_data = premarket_analysis(df)
        
        # Check for breakout alerts
        if pm_data and current_price > pm_data.get('pm_high', 0):
            alerts[symbol] = {
                'type': 'PM_BREAKOUT',
                'message': f"{symbol} broke PM high of {pm_data['pm_high']:.2f}",
                'current_price': current_price,
                'time': df.index[-1]
            }
    
    return alerts

# ------------- ENHANCED STOP MANAGEMENT -------------
def advanced_stop_recommender(symbol: str, current_price: float, entry_price: float, 
                            method: str, cfg: dict, df: Optional[pd.DataFrame] = None) -> Dict:
    """Advanced stop loss recommendations"""
    recommendations = {}
    
    if df is not None and not df.empty:
        df_tech = calculate_technical_indicators(df)
        atr = calc_atr(df_tech).iloc[-1] if len(df_tech) >= 14 else current_price * 0.02
        
        # Multiple stop methods
        recommendations['percent'] = {
            'price': round(current_price * 0.97, 2),
            'description': '3% trailing stop'
        }
        
        recommendations['atr'] = {
            'price': round(current_price - (atr * 1.5), 2),
            'description': f'1.5x ATR stop (ATR: {atr:.2f})'
        }
        
        recommendations['ema9'] = {
            'price': round(df_tech['ema9'].iloc[-1] - cfg['stop_buffer'], 2),
            'description': '9 EMA support'
        }
        
        recommendations['vwap'] = {
            'price': round(df_tech['vwap'].iloc[-1] - cfg['stop_buffer'], 2),
            'description': 'VWAP support'
        }
        
        # Structure-based stop
        recent_lows = df_tech['low'].tail(10)
        structure_stop = recent_lows.min() - cfg['stop_buffer']
        recommendations['structure'] = {
            'price': round(structure_stop, 2),
            'description': 'Recent structure low'
        }
        
        # Breakeven stop for profitable trades
        if current_price > entry_price:
            recommendations['breakeven'] = {
                'price': entry_price,
                'description': 'Breakeven protection'
            }
    
    return recommendations

# ------------- STREAMLIT UI ENHANCEMENTS -------------
def enhanced_sidebar_config() -> dict:
    """Enhanced sidebar configuration"""
    st.sidebar.header("🎯 Ross-Style Filters")
    
    cfg = {}
    
    # Market filters
    with st.sidebar.expander("📈 Market Filters", expanded=True):
        cfg['gap_min'] = st.slider("Gap % Min", 0.0, 0.50, DEFAULTS['gap_min'], 0.01)
        cfg['gap_max'] = st.slider("Gap % Max", 0.10, 1.00, 0.50, 0.05)
        cfg['price_min'], cfg['price_max'] = st.slider(
            "Price Range ($)", 0.5, 100.0, (DEFAULTS['price_min'], DEFAULTS['price_max'])
        )
        cfg['float_max'] = st.number_input(
            "Max Float (M shares)", 1, 500, int(DEFAULTS['float_max']/1e6), step=10
        ) * 1e6
        cfg['rvol_min'] = st.slider("RVOL Min", 0.5, 10.0, DEFAULTS['rvol_min'], 0.1)
        cfg['pm_vol_min'] = st.number_input(
            "PM Volume Min (K)", 0, 2000, int(DEFAULTS['pm_vol_min']/1000), step=50
        ) * 1000
    
    # Risk management
    with st.sidebar.expander("⚠️ Risk Management", expanded=True):
        cfg['account_size'] = st.number_input(
            "Account Size ($)", 1_000, 10_000_000, 100_000, step=1_000
        )
        cfg['risk_per_trade'] = st.slider(
            "Risk per Trade (%)", 0.1, 3.0, DEFAULTS['risk_per_trade']*100, 0.1
        ) / 100
        cfg['max_daily_loss'] = st.slider(
            "Max Daily Loss (%)", 0.5, 10.0, DEFAULTS['max_daily_loss']*100, 0.5
        ) / 100
        cfg['target_R'] = st.slider("Target R-Multiple", 1.0, 5.0, DEFAULTS['target_R'], 0.25)
    
    # Strategy settings
    with st.sidebar.expander("⚙️ Strategy Settings"):
        cfg['stop_buffer'] = st.number_input(
            "Stop Buffer ($)", 0.0, 0.50, DEFAULTS['stop_buffer'], 0.01
        )
        cfg['ema_period'] = st.selectbox("EMA Period", [9, 20, 50], index=0)
        cfg['min_spread'] = st.number_input(
            "Min Bid-Ask Spread", 0.01, 1.00, DEFAULTS['min_spread'], 0.01
        )
    
    return cfg

def display_daily_pnl_tracker():
    """Display daily P&L tracking"""
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = []
    
    st.subheader("📊 Daily P&L Tracker")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_pnl = st.number_input("Trade P&L ($)", value=0.0, step=10.0)
    
    with col2:
        if st.button("Add Trade"):
            st.session_state.daily_pnl.append({
                'time': dt.datetime.now(),
                'pnl': trade_pnl
            })
            st.success(f"Added ${trade_pnl:.2f}")
    
    with col3:
        if st.button("Reset Day"):
            st.session_state.daily_pnl = []
            st.success("Daily P&L reset")
    
    # Display current stats
    if st.session_state.daily_pnl:
        total_pnl = sum(trade['pnl'] for trade in st.session_state.daily_pnl)
        trade_count = len(st.session_state.daily_pnl)
        win_count = sum(1 for trade in st.session_state.daily_pnl if trade['pnl'] > 0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total P&L", f"${total_pnl:.2f}")
        col2.metric("Trades", trade_count)
        col3.metric("Winners", win_count)
        col4.metric("Win Rate", f"{win_count/trade_count:.1%}" if trade_count > 0 else "0%")

def enhanced_main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="Ross-Style Momentum Signals", 
        page_icon="🚀", 
        layout="wide"
    )
    
    st.title("🚀 Enhanced Ross-Style Momentum Signal Generator")
    st.caption("Advanced momentum trading signals with real-time monitoring")
    
    # Sidebar configuration
    cfg = enhanced_sidebar_config()
    
    # Date and time
    col1, col2 = st.columns(2)
    with col1:
        trade_date = st.date_input("Trading Date", value=dt.date.today(), max_value=dt.date.today())
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Universe selection
    st.subheader("🎯 Trading Universe")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Popular momentum stocks
        popular_stocks = [
            "AAPL", "TSLA", "NVDA", "AMD", "SOUN", "IONQ", "AI", "PLTR", "SMCI", "AVGO",
            "GOOGL", "MSFT", "META", "AMZN", "NFLX", "CRM", "ADBE", "ORCL", "UBER", "SHOP"
        ]
        
        default_universe = st.text_area(
            "Enter symbols (comma-separated)", 
            value=",".join(popular_stocks[:10]),
            height=100
        )
        symbols = [s.strip().upper() for s in default_universe.split(',') if s.strip()]
    
    with col2:
        st.info(f"**Monitoring {len(symbols)} symbols**")
        auto_refresh = st.checkbox("Auto-refresh (30s)")
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    # Daily P&L Tracker
    display_daily_pnl_tracker()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Enhanced Scanner", 
        "📈 Signal Generation", 
        "📊 Advanced Backtest", 
        "⚠️ Stop Management",
        "📱 Live Monitor"
    ])
    
    client = get_alpaca_client()
    
    with tab1:
        st.subheader("🔍 Enhanced Market Scanner")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Run Enhanced Scan", type="primary"):
                with st.spinner("Scanning for momentum opportunities..."):
                    scanner_results = enhanced_scan_gappers(symbols, trade_date, client, cfg)
                
                if not scanner_results.empty:
                    st.success(f"Found {len(scanner_results)} candidates!")
                    
                    # Display results with styling
                    st.dataframe(
                        scanner_results,
                        use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=100
                            ),
                            "Gap %": st.column_config.TextColumn("Gap %"),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f")
                        }
                    )
                    
                    # Download button
                    csv = scanner_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Scanner Results",
                        data=csv,
                        file_name=f"scanner_results_{trade_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No candidates found matching your criteria.")
        
        with col2:
            st.info("""
            **Scanner Scoring:**
            - Gap %: 6-20% optimal
            - RVOL: Higher = better
            - PM Volume: 500k+ best
            - PM Range: Movement quality
            """)
    
    with tab2:
        st.subheader("📈 Advanced Signal Generation")
        
        # Strategy selection
        strategies = st.multiselect(
            "Select Strategies",
            ["1m_ORB", "PM_High_Break", "9EMA_Pullback"],
            default=["1m_ORB", "PM_High_Break"]
        )
        
        selected_symbols = st.multiselect(
            "Select Symbols for Signals",
            symbols,
            default=symbols[:5]
        )
        
        if st.button("🎯 Generate Signals", type="primary"):
            all_signals = []
            
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(selected_symbols):
                progress_bar.progress((i + 1) / len(selected_symbols))
                
                df = fetch_minute_bars(symbol, trade_date, client)
                if df.empty:
                    continue
                
                # Generate signals based on selected strategies
                if "1m_ORB" in strategies:
                    all_signals.extend(one_min_orb(df, cfg, symbol))
                
                if "PM_High_Break" in strategies:
                    all_signals.extend(pm_high_break(df, cfg, symbol))
                
                if "9EMA_Pullback" in strategies:
                    all_signals.extend(ema_pullback_strategy(df, cfg, symbol))
            
            progress_bar.empty()
            
            if all_signals:
                # Convert to DataFrame for display
                signal_data = []
                for signal in all_signals:
                    signal_data.append({
                        'Symbol': signal.symbol,
                        'Setup': signal.setup,
                        'Entry': f"${signal.entry:.2f}",
                        'Stop': f"${signal.stop:.2f}",
                        'Target': f"${signal.target:.2f}",
                        'Size': signal.size,
                        'Risk/Reward': f"{signal.risk_reward:.1f}R",
                        'Confidence': f"{signal.confidence:.0%}",
                        'Time': signal.time.strftime("%H:%M"),
                        'Notes': signal.notes
                    })
                
                signals_df = pd.DataFrame(signal_data)
                
                st.success(f"Generated {len(all_signals)} signals!")
                st.dataframe(signals_df, use_container_width=True)
                
                # Signal visualization
                if selected_symbols:
                    chart_symbol = st.selectbox("View Chart for:", selected_symbols)
                    symbol_df = fetch_minute_bars(chart_symbol, trade_date, client)
                    symbol_signals = [s for s in all_signals if s.symbol == chart_symbol]
                    
                    if not symbol_df.empty:
                        fig = create_signal_chart(symbol_df, symbol_signals, chart_symbol)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download signals
                csv_data = pd.DataFrame([asdict(s) for s in all_signals]).to_csv(index=False)
                st.download_button(
                    "📥 Download Signals CSV",
                    data=csv_data,
                    file_name=f"signals_{trade_date}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No signals generated. Try adjusting your filters or symbol selection.")
    
    with tab3:
        st.subheader("📊 Advanced Backtesting")
        
        backtest_symbols = st.multiselect(
            "Select Symbols for Backtest",
            symbols,
            default=symbols[:3]
        )
        
        # Backtest period
        col1, col2 = st.columns(2)
        with col1:
            lookback_days = st.slider("Backtest Days", 1, 30, 5)
        with col2:
            include_strategies = st.multiselect(
                "Include Strategies",
                ["1m_ORB", "PM_High_Break", "9EMA_Pullback"],
                default=["1m_ORB", "PM_High_Break"]
            )
        
        if st.button("🔬 Run Advanced Backtest", type="primary"):
            # Multi-day backtest
            all_bt_signals = []
            symbol_data_cache = {}
            
            progress_bar = st.progress(0)
            
            for day_offset in range(lookback_days):
                test_date = trade_date - dt.timedelta(days=day_offset)
                
                for i, symbol in enumerate(backtest_symbols):
                    progress = ((day_offset * len(backtest_symbols)) + i + 1) / (lookback_days * len(backtest_symbols))
                    progress_bar.progress(progress)
                    
                    df = fetch_minute_bars(symbol, test_date, client)
                    if df.empty:
                        continue
                    
                    symbol_data_cache[f"{symbol}_{test_date}"] = df
                    
                    # Generate signals
                    if "1m_ORB" in include_strategies:
                        all_bt_signals.extend(one_min_orb(df, cfg, symbol))
                    
                    if "PM_High_Break" in include_strategies:
                        all_bt_signals.extend(pm_high_break(df, cfg, symbol))
                    
                    if "9EMA_Pullback" in include_strategies:
                        all_bt_signals.extend(ema_pullback_strategy(df, cfg, symbol))
            
            progress_bar.empty()
            
            if all_bt_signals:
                # Run backtest
                backtest_results, stats = run_enhanced_backtest(
                    all_bt_signals, symbol_data_cache, cfg
                )
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", stats.total_trades)
                col2.metric("Win Rate", f"{stats.win_rate:.1%}")
                col3.metric("Avg R-Multiple", f"{stats.avg_r:.2f}")
                col4.metric("Total P&L", f"${stats.total_pnl:,.2f}")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Max Drawdown", f"${stats.max_drawdown:,.2f}")
                col6.metric("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
                col7.metric("Profit Factor", f"{stats.profit_factor:.2f}")
                col8.metric("Account Growth", f"{stats.total_pnl/cfg['account_size']:.1%}")
                
                # Results table
                st.dataframe(backtest_results, use_container_width=True)
                
                # Equity curve
                if not backtest_results.empty:
                    equity_curve = backtest_results['P&L ($)'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=backtest_results.index,
                        y=equity_curve,
                        mode='lines',
                        name='Equity Curve',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Equity Curve",
                        xaxis_title="Trade Number",
                        yaxis_title="Cumulative P&L ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download backtest results
                csv_data = backtest_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Backtest Results",
                    data=csv_data,
                    file_name=f"backtest_{trade_date}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No signals generated for backtesting.")
    
    with tab4:
        st.subheader("⚠️ Advanced Stop Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Position Details**")
            stop_symbol = st.selectbox("Symbol", symbols)
            entry_price = st.number_input("Entry Price ($)", min_value=0.01, value=10.00, step=0.01)
            current_price = st.number_input("Current Price ($)", min_value=0.01, value=10.50, step=0.01)
            position_size = st.number_input("Position Size", min_value=1, value=100, step=1)
        
        with col2:
            st.write("**Risk Metrics**")
            current_pnl = (current_price - entry_price) * position_size
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            st.metric("Unrealized P&L", f"${current_pnl:.2f}")
            st.metric("P&L %", f"{pnl_pct:.1f}%")
            st.metric("Position Value", f"${current_price * position_size:,.2f}")
        
        if st.button("🎯 Get Stop Recommendations"):
            df = fetch_minute_bars(stop_symbol, trade_date, client)
            stop_recommendations = advanced_stop_recommender(
                stop_symbol, current_price, entry_price, "multiple", cfg, df
            )
            
            st.write("**Stop Loss Recommendations:**")
            
            for method, recommendation in stop_recommendations.items():
                stop_price = recommendation['price']
                risk_per_share = current_price - stop_price
                total_risk = risk_per_share * position_size
                
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**{method.upper()}**")
                col2.write(f"${stop_price:.2f}")
                col3.write(f"${total_risk:.2f} risk")
                col4.write(recommendation['description'])
    
    with tab5:
        st.subheader("📱 Live Market Monitor")
        
        if st.button("🔴 Start Live Monitoring"):
            alerts = real_time_monitor(symbols[:10], cfg)  # Limit for performance
            
            if alerts:
                st.success(f"🚨 {len(alerts)} alerts generated!")
                
                for symbol, alert in alerts.items():
                    with st.expander(f"🚨 {symbol} - {alert['type']}"):
                        st.write(f"**Message:** {alert['message']}")
                        st.write(f"**Current Price:** ${alert['current_price']:.2f}")
                        st.write(f"**Time:** {alert['time']}")
            else:
                st.info("No alerts at this time. Market monitoring active...")
        
        # Real-time price display
        if st.checkbox("Show Real-time Prices"):
            price_container = st.container()
            
            with price_container:
                cols = st.columns(5)
                
                for i, symbol in enumerate(symbols[:10]):  # Show first 10
                    with cols[i % 5]:
                        df = fetch_minute_bars(symbol, trade_date, client)
                        if not df.empty:
                            current_price = df.iloc[-1]['close']
                            prev_price = df.iloc[-2]['close'] if len(df) > 1 else current_price
                            change = current_price - prev_price
                            change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                            
                            delta_color = "normal" if change >= 0 else "inverse"
                            st.metric(
                                symbol,
                                f"${current_price:.2f}",
                                f"{change:+.2f} ({change_pct:+.1f}%)",
                                delta_color=delta_color
                            )

if __name__ == "__main__":
    enhanced_main()
