import os
import pandas as pd
import yfinance as yf

def download_and_split_data(symbol: str, stock_dir: str = "stock"):
    """
    Tải dữ liệu cổ phiếu từ Yahoo Finance và chia thành train/validation/test.
    Trả về dict { "train": path, "validation": path, "test": path }
    """

    os.makedirs(stock_dir, exist_ok=True)

    # Tải dữ liệu gốc (2010 - 2025)
    df = yf.download(symbol, start="2010-01-01", end="2025-01-01", interval="1d")

    if df.empty:
        return None

    # Fix MultiIndex columns nếu có
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Chia dữ liệu theo năm
    train_df = df.loc["2010-01-01":"2017-12-31"]
    validation_df = df.loc["2018-01-01":"2020-12-31"]
    test_df = df.loc["2021-01-01":"2025-01-01"]

    # Lưu ra file pickle
    paths = {}
    for split_name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        file_path = os.path.join(stock_dir, f"{symbol}_{split_name}.pkl")
        split_df.to_pickle(file_path)
        paths[split_name] = file_path

    return paths
