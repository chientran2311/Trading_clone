import os
import threading
from fastapi import FastAPI, Depends,HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from agent_api.trading_agent import TradingAgent  # import class TradingAgent
import pandas as pd

# --- Database config ---
DATABASE_URL = "postgresql://postgres:094653@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Active agents cache ---
active_agents = {}

# --- Model ---
class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    balance = Column(Float, default=10000)
    trained = Column(Boolean, default=False)   # mặc định False
    created_at = Column(DateTime, server_default=func.now())

# tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)

# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Schema ---
class AgentCreate(BaseModel):
    name: str

# --- Endpoints ---
@app.post("/agents")
def create_agent(agent: AgentCreate, db: Session = Depends(get_db)):
    new_agent = Agent(name=agent.name, balance=10000)  # trained mặc định False
    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)
    return {"message": f"Agent {new_agent.name} created successfully!", "id": new_agent.id}

@app.get("/agents")
def get_agents(db: Session = Depends(get_db)):
    agents = db.query(Agent).all()
    return agents  # trả JSON list cho frontend

@app.post("/train/{agent_id}/{symbol}")
def train_agent(agent_id: int, symbol: str, timesteps: int = 10000, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    model_path = os.path.join(MODEL_DIR, f"{agent.name}")

    if agent_id not in active_agents:
        active_agents[agent_id] = TradingAgent(agent.name, model_path=model_path, initial_capital=agent.balance)

    # --- định nghĩa function _train_and_save **trước khi gọi thread** ---
    def _train_and_save(agent_id_local, symbol_local, model_path_local, steps):
        local_db = SessionLocal()
        try:
            a = active_agents[agent_id_local]
            a.train(symbol=symbol_local, timesteps=steps)  # <-- dùng symbol
            a.save(path=model_path_local)
            row = local_db.query(Agent).filter(Agent.id == agent_id_local).first()
            if row:
                row.trained = True
                local_db.commit()
        except Exception as e:
            print("Error during training thread:", e)
        finally:
            local_db.close()

    # --- gọi thread ngay sau khi function được định nghĩa ---
    threading.Thread(
        target=_train_and_save,
        args=(agent_id, symbol, model_path, timesteps),
        daemon=True
    ).start()

    return {
        "success": True,
        "agent_id": agent.id,
        "message": f"Training started for Agent {agent.name}. Model will be saved to {model_path}.zip"
    }


@app.get("/portfolio")
def get_portfolio(agent_id: int, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        return {"success": False, "message": f"Agent {agent_id} not found!"}

    # load model into cache if trained and model file exists
    if agent_id not in active_agents and agent.trained:
        model_path = os.path.join(MODEL_DIR, f"{agent.name}")
        if os.path.exists(model_path + ".zip"):
            active_agents[agent_id] = TradingAgent(agent.name, model_path=model_path, initial_capital=agent.balance)

    return agent