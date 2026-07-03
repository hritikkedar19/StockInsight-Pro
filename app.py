import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockInsight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #161B22; }
    h1, h2, h3 { color: #FFFFFF !important; }
    div.stMetric { background-color: #21262D; padding: 15px; border-radius: 10px; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; border: none; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'high_52w': info.get('fiftyTwoWeekHigh', 0),
            'low_52w': info.get('fiftyTwoWeekLow', 0),
        }
    except:
        return None

def create_features(df):
    data = df.copy()
    if 'Close' not in data.columns or len(data) < 60:
        return None
    
    # Simple features
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Target: Next day close
    data['Target'] = data['Close'].shift(-1)
    
    return data.dropna()

def predict_next_price(data):
    try:
        # Get last valid row for prediction
        df = data.copy()
        
        # Features to use
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'Price_Change']
        
        # Make sure all columns exist
        for f in features:
            if f not in df.columns:
                return None, None
        
        X = df[features]
        y = df['Target']
        
        if len(X) < 50:
            return None, None
        
        # Use last 100 rows for more stable training
        train_size = min(100, len(X) - 10)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_last = X.iloc[-1:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Predict
        prediction = model.predict(X_last)[0]
        actual = df['Close'].iloc[-1]
        
        return float(prediction), float(actual)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 StockInsight Pro")
    st.markdown("---")
    
    stocks = {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.", 
        "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation", "NFLX": "Netflix Inc.", "AMD": "Advanced Micro Devices",
        "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America", "WFC": "Wells Fargo",
        "V": "Visa Inc.", "MA": "Mastercard", "WMT": "Walmart Inc.", "HD": "Home Depot",
        "NKE": "Nike Inc.", "SBUX": "Starbucks Corp.", "MCD": "McDonald's Corp.",
        "KO": "Coca-Cola Company", "PEP": "PepsiCo Inc.", "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group", "PFE": "Pfizer Inc.", "XOM": "Exxon Mobil",
        "CVX": "Chevron Corporation", "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust",
    }
    
    sorted_stocks = dict(sorted(stocks.items()))
    selected_ticker = st.selectbox("Select Stock", list(sorted_stocks.keys()), index=0)
    company_name = sorted_stocks[selected_ticker]
    
    st.markdown("---")
    st.markdown("### 📅 Date Range")
    
    # Default to last 2 years
    default_start = datetime.now() - timedelta(days=730)
    start_date = st.date_input("From", default_start)
    end_date = st.date_input("To", datetime.now())
    
    st.markdown("---")
    st.caption("📈 StockInsight Pro | Data: Yahoo Finance")
    st.caption("⚠️ For educational purposes only.")

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
data = get_stock_data(selected_ticker, start_date, end_date)

if data is None or data.empty:
    st.error(f"Unable to load data for {selected_ticker}")
    st.stop()

if len(data) < 30:
    st.error("Not enough data. Please select a longer date range.")
    st.stop()

info = get_stock_info(selected_ticker)
processed_data = create_features(data)

col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"{company_name}")
    st.markdown(f"**{selected_ticker}**")
    if info and info.get('sector'):
        st.caption(f"📂 {info['sector']} • 🔧 {info['industry']}")
with col2:
    current_price = float(data['Close'].iloc[-1])
    price_change = current_price - float(data['Close'].iloc[0])
    price_change_pct = (price_change / float(data['Close'].iloc[0])) * 100
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔮 Prediction", "ℹ️ Info"])

# -------- CHARTS TAB --------
with tab1:
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
    with col_opt2:
        show_ma = st.checkbox("Show Moving Averages", value=True)
    
    if chart_type == "Candlestick":
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350'
        ))
        if show_ma:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(), name='20-Day MA', line=dict(color='#FF5722', width=1.5)))
        fig.update_layout(template="plotly_dark", title={'text': f'{company_name} - Price Chart', 'x': 0.5, 'font': dict(size=20)}, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,27,34,0.8)', xaxis=dict(showgrid=True, gridcolor='#303632'), yaxis=dict(showgrid=True, gridcolor='#303632'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=50, r=50, t=80, b=50))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='#00E676', width=2), fill='tozeroy', fillcolor='rgba(0,230,118,0.1)'))
        if show_ma:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(), name='20-Day MA', line=dict(color='#FF5722', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), name='50-Day MA', line=dict(color='#2196F3', width=1.5, dash='dot')))
        fig.update_layout(template="plotly_dark", title={'text': f'{company_name} - Price Trend', 'x': 0.5, 'font': dict(size=20)}, height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,27,34,0.8)', margin=dict(l=50, r=50, t=80, b=50))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Trading Volume")
    vol_colors = ['#26A69A' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#EF5350' for i in range(len(data))]
    fig_vol = go.Figure(go.Bar(x=data.index, y=data['Volume'], marker_color=vol_colors, opacity=0.8))
    fig_vol.update_layout(template="plotly_dark", title={'text': 'Trading Volume', 'x': 0.5}, height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,27,34,0.8)', showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

# -------- PREDICTION TAB --------
with tab2:
    st.markdown("### 🔮 Next Day Price Forecast")
    st.caption("Powered by Machine Learning (Random Forest)")
    
    if processed_data is not None:
        pred_result = predict_next_price(processed_data)
        
        if pred_result[0] is not None:
            pred_price, actual_price = pred_result
            
            p1, p2, p3 = st.columns(3)
            with p1: 
                st.info(f"**Current Price**\n\n### ${actual_price:.2f}")
            with p2: 
                st.success(f"**Predicted Tomorrow**\n\n### ${pred_price:.2f}")
            with p3:
                change = pred_price - actual_price
                change_pct = (change / actual_price) * 100
                st.metric("Expected Change", f"${change:.2f}", f"{change_pct:.2f}%")
            
            st.markdown("---")
            
            # Recommendation
            recommendation = "🟢 BUY" if pred_price > actual_price else "🔴 SELL"
            rec_color = "#26A69A" if pred_price > actual_price else "#EF5350"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: #21262D; border-radius: 15px;'>
                <h3 style='color: white; margin-bottom: 10px;'>🤖 AI Analyst Recommendation</h3>
                <h1 style='color: {rec_color}; margin: 0;'>{recommendation}</h1>
                <p style='color: #8B949E; margin-top: 10px;'>Based on historical pattern analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Unable to generate prediction. Not enough data.")
    else:
        st.warning("Processing data... Please select a longer date range (at least 2 years recommended).")

# -------- INFO TAB --------
with tab3:
    if info:
        m1, m2, m3, m4 = st.columns(4)
        mcap = f"${info['market_cap']/1e12:.2f}T" if info['market_cap'] > 1e12 else f"${info['market_cap']/1e9:.2f}B" if info['market_cap'] > 1e9 else "N/A"
        m1.metric("Market Cap", mcap)
        m2.metric("P/E Ratio", f"{info['pe_ratio']:.2f}" if info['pe_ratio'] else "N/A")
        m3.metric("52W High", f"${info['high_52w']:.2f}")
        m4.metric("52W Low", f"${info['low_52w']:.2f}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About This Stock")
        st.write(f"**{company_name}** ({selected_ticker})")
        st.write(f"Sector: **{info['sector']}**")
        st.write(f"Industry: **{info['industry']}**")
        
        st.markdown("### 📈 Key Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Average Volume', 'Current Volume', 'Open', 'Previous Close'], 
            'Value': [
                f"{info['avg_volume']:,}" if info['avg_volume'] else "N/A",
                f"{info['volume']:,}" if info['volume'] else "N/A",
                f"${data['Open'].iloc[-1]:.2f}",
                f"${data['Close'].iloc[-2]:.2f}" if len(data) > 1 else "N/A"
            ]
        })
        st.table(stats_df)

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #8B949E; padding: 20px;'><p>📈 StockInsight Pro | Data provided by Yahoo Finance</p><p style='font-size: 12px;'>⚠️ For educational purposes only. Not financial advice.</p></div>", unsafe_allow_html=True)