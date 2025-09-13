# 🚀 Trading Agent Project

Dự án này xây dựng một **Trading Agent** sử dụng Reinforcement Learning (RL) với thư viện **Stable-Baselines3 (PPO)**, kết hợp với **FastAPI** để làm API và **Streamlit** để trực quan hóa dữ liệu & giao diện người dùng.

---

## 🛠️ Công nghệ & Thư viện chính

- **Python 3.12**
- **NumPy** – xử lý dữ liệu số  
- **Pandas** – đọc & phân tích dữ liệu  
- **Gymnasium** – xây dựng môi trường RL  
- **Stable-Baselines3** – RL (PPO)  
- **gym-anytrading** – môi trường trading  
- **FastAPI** – API phục vụ train/evaluate  
- **SQLAlchemy** – ORM quản lý database  
- **Streamlit** – UI phân tích & visualization  
- **Plotly** – vẽ biểu đồ tương tác  
- **yfinance** – tải dữ liệu chứng khoán  

---

## ⚙️ Cài đặt môi trường

### 1️⃣ Clone dự án
```bash
git clone https://github.com/username/Trading_clone.git
cd Trading_clone
```

### 2️⃣ Tạo virtual environment
```bash
python -m venv .venv
```

**Kích hoạt môi trường ảo:**  
- **Mac/Linux:**
```bash
source .venv/bin/activate
```
- **Windows:**
```bash
.venv\Scripts\activate
```

### 3️⃣ Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Cách chạy dự án

### 🔹 1. Chạy API (backend)
- Chuẩn bị dữ liệu: đặt file vào thư mục `stock/{symbol}_train.pkl`  
  ⚠️ File này **bắt buộc phải có cột `Close`**

Chạy FastAPI server:
```bash
uvicorn main:app --reload
```

### 🔹 2. Chạy Streamlit (UI)
```bash
streamlit run pages/3_StockScreener.py
```

### 🔹 3. Enjoy 🚀
Truy cập API & giao diện để bắt đầu thử nghiệm agent trading.

---

## 📩 Liên hệ
Nếu có bất kỳ vấn đề nào, vui lòng liên hệ:  
📧 **chien2977@gmail.com**
