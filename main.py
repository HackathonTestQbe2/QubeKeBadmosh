import pandas as pd
import numpy as np
import datetime
import os, sys
import importlib
import utils
importlib.reload(utils)
from utils import plot_series, plot_series_with_names, plot_series_bar
from utils import plot_dataframe
from utils import get_universe_adjusted_series, scale_weights_to_one, scale_to_book_long_short
from utils import backtest, metrics_from_holdings
import plotly.graph_objects as go

# Loading features, returns and universe from the `phase{}` directory

# Change the directory as per your requirements
data_dir = "/kaggle/input/qrt-quant-dataquest-2025-iit-delhi/Validation"

features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))

universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))

# Start writing the `get_weights` function which contains your strategy code

# A Benchmark Strategy for your reference: 
# This is the code used to generate the Benchmark submission you see in the Kaggle Leaderboard

import pandas as pd
import numpy as np
import datetime
import os, sys
import importlib
import utils
importlib.reload(utils)
from utils import plot_series, plot_series_with_names, plot_series_bar
from utils import plot_dataframe
from utils import get_universe_adjusted_series, scale_weights_to_one, scale_to_book_long_short
from utils import backtest, metrics_from_holdings
import plotly.graph_objects as go

# Loading features, returns and universe from the `phase{}` directory

# Change the directory as per your requirements
data_dir = "/kaggle/input/qrt-quant-dataquest-2025-iit-delhi/Validation"

features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))

universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))

# Start writing the `get_weights` function which contains your strategy code

# A Benchmark Strategy for your reference: 
# This is the code used to generate the Benchmark submission you see in the Kaggle Leaderboard

# This strategy shows how you can combine different features 
def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    
    """
    Calculate stock weights for the portfolio on the current trading day.

    Parameters:
    -----------
    features : pd.DataFrame
        A pandas DataFrame containing feature data.
        - Index: Datetime (chronological order).
        - Columns: MultiIndex with two levels:
          - Level 0: Feature names (e.g., "f1", "f2", ..., "f25").
          - Level 1: Stock identifiers (e.g., "1", "2", ..., "2167").

    today_universe : pd.Series
        A pandas Series indicating the stock universe for the current day.
        - Index: Stock identifiers.
        - Values: 0 or 1, where 1 means the stock is in the universe and 
          0 means it is not in the universe.

    Returns:
    --------
    dict[str, float]
        A dictionary where:
        - Keys: Stock identifiers (strings).
        - Values: Weights/positions of the stocks (floats) for the current trading day.
    """
    if features.shape[0] == 0:
        return {}
           
    alpha_signal = features["f22"].iloc[-1]  # Get the latest day's values

    # Ensure alpha_signal is a Series (Fixing IndexError)
    if not isinstance(alpha_signal, pd.Series):
        alpha_signal = pd.Series(alpha_signal, index=today_universe.index)

    # Apply universe constraint (only tradable stocks)
    alpha_signal = alpha_signal[today_universe.astype(bool)]

    # Rank stocks (higher values = stronger buy signal)
    ranked_alpha = alpha_signal.rank(method="min", ascending=True)

    # Normalize: Mean-zero for dollar neutrality
    ranked_alpha = ranked_alpha.sub(ranked_alpha.mean())

    # Scale weights to ensure sum(|weights|) ≤ 1
    final_weights = ranked_alpha.div(ranked_alpha.abs().sum())

    # Convert to dictionary format {Stock ID: Weight}
    return final_weights.to_dict()



# Backtest your strategy to see how it performed

positions, sr = backtest(
    get_weights,
    features,
    returns,
    universe,
    "2005-01-03",
    "2019-12-31",
    True,
    True
)

# Submit this csv file on kaggle

positions.to_csv("submission.csv")

# Test your generated portfolio positions

nsr = metrics_from_holdings(positions, returns, universe)
