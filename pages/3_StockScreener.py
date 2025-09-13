import streamlit as st
import plotly.express as px
import pandas as pd
import os

from ETL.transdata import download_and_split_data  # hàm tải & chia data
from agent_api.trading_agent import TradingAgent   # import agent thật

st.set_page_config(page_title="Stock Screener", layout="wide")

# ===== Hiển thị số dư góc phải =====
def show_balance(agent):
    st.markdown(
        f"""
        <div style="position:fixed; top:10px; right:20px; background:#f0f2f6;
                    padding:8px 16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.2);">
            💰 <b>Agent</b>: {agent.capital:,.2f} USD
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Main UI =====
st.title("📈 Stock Screener")

symbol = st.text_input("Nhập mã cổ phiếu", "AAPL").upper()
stock_dir = "stock"
os.makedirs(stock_dir, exist_ok=True)

# ===== Session state =====
if "data_paths" not in st.session_state:
    st.session_state.data_paths = None
if "dataset_option" not in st.session_state:
    st.session_state.dataset_option = "Train"
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None

get_data = st.button("Lấy dữ liệu")

# Hàm hiển thị chart
def show_chart(df, title):
    if df is not None and not df.empty:
        fig = px.line(
            df,
            x=df.index,
            y="Close",
            title=title
        )
        fig.update_traces(mode="lines+markers", hovertemplate="Ngày: %{x}<br>Giá: %{y} USD")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail())
    else:
        st.warning("❌ Không có dữ liệu để hiển thị.")

# Khi bấm Lấy dữ liệu
if get_data:
    paths = download_and_split_data(symbol, stock_dir)
    if paths is None:
        st.warning("❌ Không tìm thấy dữ liệu cho mã này.")
    else:
        st.success(f"✅ Đã tải và chia dữ liệu thành train/validation/test cho {symbol}")
        st.session_state.data_paths = paths

# Nếu đã có dữ liệu thì hiển thị
if st.session_state.data_paths:
    # Select box chọn loại dữ liệu
    option = st.selectbox(
        "Chọn dữ liệu để hiển thị:",
        ["Train", "Validation", "Test"],
        index=["Train", "Validation", "Test"].index(st.session_state.dataset_option),
        key="dataset_option"
    )

    file_map = {
        "Train": f"{symbol}_train.pkl",
        "Validation": f"{symbol}_validation.pkl",
        "Test": f"{symbol}_test.pkl"
    }
    path = os.path.join(stock_dir, file_map[option])
    df = pd.read_pickle(path) if os.path.exists(path) else None
    show_chart(df, f"{symbol} {option} Set")

    # ===== Lấy danh sách agent từ folder models/ =====
    # --- Lấy danh sách agent từ folder models/ ---
    model_dir = "models"
    available_agents = [
        f.replace(".zip", "") for f in os.listdir(model_dir) if f.endswith(".zip")
] if os.path.exists(model_dir) else []

    if not available_agents:
        st.warning("⚠️ Chưa có agent nào trong thư mục models/.")
        selected_agent = None
    else:
        selected_agent = st.selectbox(
            "Chọn agent đã train:",
            available_agents,
            index=available_agents.index(st.session_state.selected_agent)
              if st.session_state.selected_agent in available_agents else 0,
            key="selected_agent"
    )
        st.info(f"Bạn đã chọn agent: {selected_agent}")

    # --- Hai nút Validation & Test ---
        # --- Hai nút Validation & Test ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Validation"):
        if selected_agent is None:
            st.warning("⚠️ Vui lòng chọn agent trước")
        else:
            agent_path = os.path.join(model_dir, f"{selected_agent}.zip")
            agent = TradingAgent(
                name=selected_agent,
                model_path=agent_path,
                transaction_fee=st.session_state.get("transaction_fee", 0.001),
                slippage=st.session_state.get("slippage", 0.001)
            )

            # Gọi evaluate trực tiếp, tự load dữ liệu validation dựa vào symbol
            results, trade_log, equity_curve = agent.evaluate(symbol, mode="validation", return_log=True)

            # Buy & Hold baseline
            df_val = pd.read_pickle(os.path.join(stock_dir, f"{symbol}_validation.pkl"))
            prices = df_val["Close"].tolist()
            initial_capital = 10000
            buy_hold_final = initial_capital * (prices[-1] / prices[0])
            buy_hold_profit = buy_hold_final - initial_capital

            # Show metrics
            st.success(f"✅ Validation {selected_agent}")
            st.markdown(f"""
**Agent Performance**
- Profit: {results['profit']:.2f} USD
- Loss: {results['loss']:.2f} USD
- Final Capital: {results['final_capital']:.2f} USD
- Initial Investment: {results['initial_capital']:.2f} USD
- Total Trades: {results['total_trades']}
- Buy: {results['buy_count']} | Sell: {results['sell_count']} | Hold: {results['hold_count']}
""")

            st.markdown(f"""
**Buy & Hold Baseline**
- Profit: {buy_hold_profit:.2f} USD
- Final Capital: {buy_hold_final:.2f} USD
""")

            st.line_chart(equity_curve, height=300)
            st.dataframe(trade_log)

with col2:
    if st.button("🧪 Test"):
        if selected_agent is None:
            st.warning("⚠️ Vui lòng chọn agent trước")
        else:
            agent_path = os.path.join(model_dir, f"{selected_agent}.zip")
            agent = TradingAgent(
                name=selected_agent,
                model_path=agent_path,
                transaction_fee=st.session_state.get("transaction_fee", 0.001),
                slippage=st.session_state.get("slippage", 0.001)
            )

            # Gọi evaluate trực tiếp, tự load dữ liệu test dựa vào symbol
            results, trade_log, equity_curve = agent.evaluate(symbol, mode="test", return_log=True)

            # Buy & Hold baseline
            df_test = pd.read_pickle(os.path.join(stock_dir, f"{symbol}_test.pkl"))
            prices = df_test["Close"].tolist()
            initial_capital = 10000
            buy_hold_final = initial_capital * (prices[-1] / prices[0])
            buy_hold_profit = buy_hold_final - initial_capital

            # Show metrics
            st.success(f"✅ Test {selected_agent}")
            st.markdown(f"""
**Agent Performance**
- Profit: {results['profit']:.2f} USD
- Loss: {results['loss']:.2f} USD
- Final Capital: {results['final_capital']:.2f} USD
- Initial Investment: {results['initial_capital']:.2f} USD
- Total Trades: {results['total_trades']}
- Buy: {results['buy_count']} | Sell: {results['sell_count']} | Hold: {results['hold_count']}
""")

            st.markdown(f"""
**Buy & Hold Baseline**
- Profit: {buy_hold_profit:.2f} USD
- Final Capital: {buy_hold_final:.2f} USD
""")

            st.line_chart(equity_curve, height=300)
            st.dataframe(trade_log)
