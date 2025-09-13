import streamlit as st
import pandas as pd
import requests
import os

st.title("💼 Portfolio Manager")

API_URL = "http://127.0.0.1:8000"

# --- Tạo agent mới ---
with st.expander("➕ Create Agent"):
    new_name = st.text_input("Tên Agent", "")
    if st.button("Create Agent"):
        if new_name.strip() != "":
            try:
                res = requests.post(f"{API_URL}/agents", json={"name": new_name})
                if res.status_code == 200:
                    st.success(f"✅ Agent '{new_name}' đã được tạo")
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("❌ Vui lòng nhập tên Agent")

# --- Lấy danh sách agents ---
agents = []
try:
    res = requests.get(f"{API_URL}/agents")
    if res.status_code == 200:
        agents = res.json()
except Exception as e:
    st.error(f"Lỗi khi load agents: {e}")

# --- Selectbox chọn agent ---
agent_id = None
if agents:
    df_agents = pd.DataFrame(agents)
    st.subheader("📋 Danh sách Agents")
    st.dataframe(df_agents[["id", "name", "balance", "trained", "created_at"]])

    agent_names = {a["name"]: a["id"] for a in agents}
    chosen_name = st.selectbox("Chọn Agent", list(agent_names.keys()))
    agent_id = agent_names[chosen_name]

# --- Selectbox chọn symbol từ folder stock ---
stock_dir = "stock"
symbols = [f.replace(".pkl","") for f in os.listdir(stock_dir) if f.endswith(".pkl")]
symbol = None
if symbols:
    symbol = st.selectbox("Chọn mã cổ phiếu", symbols)

# --- Nút train agent ---
if st.button("⚡ Train Agent"):
    if agent_id and symbol:
        try:
            res = requests.post(f"{API_URL}/train/{agent_id}/{symbol}")
            if res.status_code == 200:
                st.success("Agent trained successfully!")
                st.json(res.json())
            else:
                st.error(f"Error: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("⚠️ Chưa chọn Agent hoặc mã cổ phiếu")
