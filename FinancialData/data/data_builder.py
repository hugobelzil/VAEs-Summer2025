import pandas as pd
import numpy as np
from functools import reduce
import os
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1) Map tickers to the filenames you uploaded
# ------------------------------------------------------------------
files = {
    "DUK":  "duke2005.csv",       # Duke Energy
    "BA":   "boeing2005.csv",     # Boeing
    "NVDA": "nvidia2005.csv",     # Nvidia
    "KO":   "coca2005.csv",       # Coca-Cola
    "MSFT": "msft2005.csv",       # Microsoft
    "NFLX": "netflix2005.csv"     # Netflix
}

# Quick sanity-check: make sure every file is really there
for f in files.values():
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found in the current directory.")


# ------------------------------------------------------------------
# 2) Load, keep (Date, Close), compute log-returns ln(P_t / P_{t-1})
# ------------------------------------------------------------------
dfs = []
for ticker, fname in files.items():
    df = (
        pd.read_csv(fname, parse_dates=["Date"])
          .loc[:, ["Date", "Close"]]
          .sort_values("Date")
          .rename(columns={"Close": ticker})
    )
    df[ticker] = np.log(df[ticker]).diff()   # one-day log return
    dfs.append(df)

# ------------------------------------------------------------------
# 3) Inner-join on shared trading dates, drop NaNs from first diff
# ------------------------------------------------------------------
returns = reduce(lambda left, right:
                 pd.merge(left, right, on="Date", how="inner"), dfs)
returns = returns.dropna().set_index("Date")


# ------------------------------------------------------------------
# 4) Six-dimensional matrix for the VAE
# ------------------------------------------------------------------
X = returns.values          # shape: (n_days, 6)

print(returns.head()) # preview with dates
X = 10*X #WORKING WITH 10*LOG RETURNS
X = X.astype("float32")
print(X.shape)              # (rows, 6)

train_dataset, eval_dataset = train_test_split(X, test_size=0.1, random_state=42)

np.save("train_financial_dataset.npy", train_dataset)
np.save("eval_financial_dataset.npy", eval_dataset)
