# 🚀 Trading Agent Project

Dự án này xây dựng một ** Trading Agent** sử dụng Reinforcement Learning (RL) với thư viện **Stable-Baselines3 (PPO)**, kết hợp với **FastAPI** để làm API và **Streamlit** để trực quan hóa và giao diện người dùng.

---

## 🛠️ Công nghệ & Thư viện chính

- **Python 3.12**
- [NumPy](https://numpy.org/) – xử lý dữ liệu số
- [Pandas](https://pandas.pydata.org/) – đọc & phân tích dữ liệu
- [Gymnasium](https://gymnasium.farama.org/) – xây dựng môi trường RL
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) – RL (PPO)
- [gym-anytrading](https://pypi.org/project/gym-anytrading/) – môi trường trading
- [FastAPI](https://fastapi.tiangolo.com/) – API phục vụ train/evaluate
- [SQLAlchemy](https://www.sqlalchemy.org/) – ORM quản lý database
- [Streamlit](https://streamlit.io/) – UI phân tích & visualization
- [Plotly](https://plotly.com/python/) – vẽ biểu đồ tương tác
- [yfinance](https://pypi.org/project/yfinance/) – tải dữ liệu chứng khoán

---

## ⚙️ Cài đặt môi trường

### 1️⃣ Clone dự án

```bash
git clone https://github.com/username/Trading_clone.git
cd Trading_clone

### 2️⃣ Tạo virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

### 3️⃣ Cài đặt thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt

📊 Cách chạy dự án
🔹 chạy api (backend)
Chuẩn bị dữ liệu stock/{symbol}_train.pkl (cột Close bắt buộc).
Chạy API train:
``` uvicorn main:app --reload
🔹 2. Chạy streamlit (hiển thị giao diện)
``` streamlit run pages/3_StockScreener.py
🔹 3. enjoy

if there is any problems pls contact chien2977@gmail.com
