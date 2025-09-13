import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import gym_anytrading

class TradingEnv(gym.Env):
       
        def __init__(self, prices, initial_capital=10000,
                    transaction_fee=0.001, slippage=0.001,
                    max_position=100, unit_size=1):
            super().__init__()
            self.prices = list(prices)
            self.n_steps = len(self.prices)
            self.initial_capital = float(initial_capital)
            self.transaction_fee = float(transaction_fee)
            self.slippage = float(slippage)
            self.max_position = int(max_position)
            self.unit_size = int(unit_size)
            self.window_size = 5  # hoặc tuỳ bạn muốn dùng 10 ngày

            self.action_space = gym.spaces.Discrete(3)  # 0=hold,1=buy,2=sell
            self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.window_size+2,), dtype=np.float32)

            self.reset()

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.capital = self.initial_capital
            self.shares = 0
            self.step_idx = 0
            self.prev_value = self.initial_capital
            self.buy_count = self.sell_count = self.hold_count = 0
            self.actions_log = []
            start_idx = 0
            window_prices = self.prices[start_idx:start_idx + self.window_size]
            obs = np.array(window_prices + [self.capital, self.shares], dtype=np.float32)
            info = {"capital": self.capital, "shares": self.shares, "total_value": self.capital}
            return obs, info

        def step(self, action):
            price = float(self.prices[self.step_idx])
            buy_price = price * (1 + self.slippage)
            sell_price = price * (1 - self.slippage)
            start_idx = max(0, self.step_idx - self.window_size + 1)
            window_prices = self.prices[start_idx:self.step_idx + 1]
            # pad nếu thiếu
            if len(window_prices) < self.window_size:
                window_prices = [self.prices[0]]*(self.window_size - len(window_prices)) + window_prices
            obs = np.array(window_prices + [self.capital, self.shares], dtype=np.float32)

            # BUY
            if action == 1:
                qty = self.unit_size
                if (self.shares + qty) <= self.max_position and self.capital >= buy_price * qty * (1 + self.transaction_fee):
                    cost = buy_price * qty * (1 + self.transaction_fee)
                    self.capital -= cost
                    self.shares += qty
                    self.buy_count += 1
                else:
                    self.hold_count += 1

            # SELL
            elif action == 2:
                qty = self.unit_size
                if self.shares >= qty:
                    proceeds = sell_price * qty * (1 - self.transaction_fee)
                    self.capital += proceeds
                    self.shares -= qty
                    self.sell_count += 1
                else:
                    self.hold_count += 1
            else:
                self.hold_count += 1

            total_value = self.capital + self.shares * price
            reward = 100 * (total_value - self.prev_value) / self.prev_value
            self.prev_value = total_value

            obs = np.array(window_prices + [self.capital, self.shares], dtype=np.float32)
            self.actions_log.append((self.step_idx, int(action), price, float(total_value)))

            self.step_idx += 1
            done = self.step_idx >= (self.n_steps - 1)

            info = {"capital": self.capital, "shares": self.shares, "total_value": total_value}
            return obs, float(reward), bool(done), False, info


class TradingAgent:
        def __init__(self, name, model_path=None, transaction_fee=0.001,
                    slippage=0.001, initial_capital=10000):
            self.name = name
            self.model_path = model_path
            self.transaction_fee = transaction_fee
            self.slippage = slippage
            self.initial_capital = float(initial_capital)
            self.model = None
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)

        def make_env(self, prices=None, symbol=None, max_position=100, unit_size=1):
            if symbol:
                file_path = os.path.join("stock", f"{symbol}.pkl")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Không tìm thấy dữ liệu {file_path}")

                df = pd.read_pickle(file_path)
                if "Close" not in df.columns:
                    raise ValueError("File không có cột 'Close'")

                prices = df["Close"].tolist()
                return TradingEnv(
            prices,
            initial_capital=self.initial_capital,
            transaction_fee=self.transaction_fee,
            slippage=self.slippage,
            max_position=max_position,
            unit_size=unit_size
        )

            elif prices:
                return TradingEnv(
            prices,
            initial_capital=self.initial_capital,
            transaction_fee=self.transaction_fee,
            slippage=self.slippage,
            max_position=max_position,
            unit_size=unit_size
        )
            else:
                raise ValueError("Cần truyền prices hoặc symbol để tạo môi trường")



        def train(self, symbol: str, timesteps=10000):
            stock_dir = "stock"
            file_path = os.path.join(stock_dir, f"{symbol}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file {file_path}")

            df = pd.read_pickle(file_path)
            if "Close" not in df.columns:
                raise ValueError(f"File {file_path} không có cột 'Close'")

            prices = df["Close"].tolist()

    # ✅ Dùng trực tiếp TradingEnv (đồng bộ với evaluate)
            env = TradingEnv(
        prices,
        initial_capital=self.initial_capital,
        transaction_fee=self.transaction_fee,
        slippage=self.slippage
    )

            self.model = PPO("MlpPolicy", env, verbose=1)
            self.model.learn(total_timesteps=timesteps)
            env.close()

    # ✅ Đảm bảo luôn lưu model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{symbol}_agent.zip")

            self.model.save(model_path)
            self.model_path = model_path  # ghi nhớ đường dẫn để load sau
            print(f"✅ Model saved at {model_path}")



        def save(self, path=None):
            if not path:
                path = self.model_path
            if self.model:
                self.model.save(path)
                print(f"✅ Model saved at {path}")

        def load_model(self, path):
            self.model = PPO.load(path)
            print(f"🔄 Model loaded from {path}")

        def act(self, state):
            if self.model:
        # Đảm bảo state luôn có shape (1, obs_dim)
                if state.ndim == 1:
                    state = state.reshape(1, -1)
                action, _ = self.model.predict(state, deterministic=True)
                return int(action)
            return 0

        def evaluate(self, symbol, mode="validation", return_log=False):
  
            stock_dir = "stock"
            file_map = {
            "validation": f"{symbol}_validation.pkl",
            "test": f"{symbol}_test.pkl"
         }

            if mode not in file_map:
                raise ValueError("Mode phải là 'validation' hoặc 'test'")

            file_path = os.path.join(stock_dir, file_map[mode])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file {file_path}")

            df = pd.read_pickle(file_path)
            if "Close" not in df.columns:
                raise ValueError(f"File {file_path} không có cột 'Close'")

            prices = df["Close"].tolist()
    
    # Tạo môi trường TradingEnv
            env = TradingEnv(prices,
                     initial_capital=self.initial_capital,
                     transaction_fee=self.transaction_fee,
                     slippage=self.slippage)
    
            obs, info = env.reset()
            done, truncated = False, False
            trade_log = []

            while not (done or truncated):
                action = self.act(obs)
                obs, reward, done, truncated, info = env.step(action)
                trade_log.append({**info, "action": action, "price": obs[0]})

            final_capital = info["total_value"]
            profit = final_capital - self.initial_capital
            loss = -profit if profit < 0 else 0
            total_trades = env.buy_count + env.sell_count + env.hold_count

            results = {
        "profit": profit,
        "loss": loss,
        "final_capital": final_capital,
        "initial_capital": self.initial_capital,
        "buy_count": env.buy_count,
        "sell_count": env.sell_count,
        "hold_count": env.hold_count,
        "total_trades": total_trades
    }

            trade_log_df = pd.DataFrame(trade_log)
            equity_curve = trade_log_df["total_value"].tolist() if not trade_log_df.empty else []

            if return_log:
                return results, trade_log_df, equity_curve
            return results
