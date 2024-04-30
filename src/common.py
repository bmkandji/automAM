import numpy as np
import pandas as pd
def compute_log_returns(df: pd.DataFrame, scale: float = 100) -> pd.DataFrame:
    """
    Calculates the logarithmic returns for the adjusted close prices in the DataFrame and scales them by a specified factor.

    Parameters:
    scale (float): The factor to scale the log returns, default is 100 (to represent log returns in percentage terms).

    Returns:
    pd.DataFrame: A DataFrame containing the scaled logarithmic returns.
    """
    # Calculate logarithmic returns using NumPy's log function
    log_returns = np.log(df / df.shift(1)) * scale

    # Remove the first date row to avoid NaN values resulting from the shift operation
    log_returns = log_returns.iloc[1:]  # Excludes the first row

    return log_returns

