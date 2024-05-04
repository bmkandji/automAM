import numpy as np
import pandas as pd
from typing import Any, Optional


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


def weighting(data: np.ndarray, scheme: str = 'uniform', **kwargs: Any) -> np.ndarray:
    """
    Computes weighted sums of data rows based on the specified weighting scheme.

    Args:
        data (np.ndarray): 2D array where each row is a dataset.
        scheme (str): Specifies the weighting scheme to use ('uniform', 'exponential', 'gaussian', etc.).
        **kwargs: Additional parameters for specific weighting schemes, e.g., 'decay_rate' for 'exponential'.

    Returns:
        np.ndarray: Weighted sum for each row in data.

    Raises:
        ValueError: If an unsupported weighting scheme is provided.
    """

    n = data.shape[0]  # Number of data points (rows) in data

    # Define weights based on the chosen scheme
    if scheme == 'uniform':
        weights = np.ones(n) / n
    elif scheme == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.5)
        weights = np.exp(-decay_rate * np.arange(n))
        weights /= np.sum(weights)
    elif scheme == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        range_n = np.arange(n)
        weights = np.exp(-((range_n - np.mean(range_n)) ** 2) / (2 * sigma ** 2))
        weights /= np.sum(weights)
    elif scheme == 'linear':
        weights = (n - np.arange(n)) / ((n * (n + 1)) / 2)
    elif scheme == 'inverse':
        weights = 1 / (np.arange(1, n + 1))
        weights /= np.sum(weights)
    elif scheme == 'sqrt':
        weights = np.sqrt(n - np.arange(n))
        weights /= np.sum(weights)
    elif scheme == 'exponential_almon':
        omega1 = kwargs.get('omega1', 1.0)
        omega2 = kwargs.get('omega2', 0.0)
        range_n = np.arange(n)
        weights = np.exp(omega1 * range_n + omega2 * range_n ** 2)
        weights /= np.sum(weights)
    elif scheme == 'beta':
        omega1 = kwargs.get('omega1', 2.0)
        omega2 = kwargs.get('omega2', 2.0)
        weights = ((np.arange(n) / n) ** (omega1 - 1)) * ((1 - np.arange(n) / n) ** (omega2 - 1))
        weights /= np.sum(weights)
    elif scheme == 'exponential_simple':
        omega = kwargs.get('omega', 0.5)
        weights = omega ** np.arange(n)
        weights /= np.sum(weights)
    else:
        raise ValueError("Unsupported weighting scheme")

    # Calculate the weighted average of the matrices along the first axis (rows),
    # using the specified weights to compute the average.
    weighted_mean = np.average(data, axis=0, weights=weights)

    return weighted_mean
