import numpy as np
import pandas as pd
from typing import Any
import exchange_calendars as ecals
from datetime import datetime, timedelta


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
        raise ValueError("Unsupported weightsing scheme")

    # Calculate the weightsed average of the matrices along the first axis (rows),
    # using the specified weights to compute the average.
    weightsed_mean = np.average(data, axis=0, weights=weights)

    return weightsed_mean


def get_last_trading_day(given_date, market):
    """
    Returns the last trading day for the NASDAQ market before the given date.

    :param given_date: The given date as a datetime object.
    :return: The last trading day as a datetime object.

    Args:
        market:
    """
    # NASDAQ calendar
    market_cal = ecals.get_calendar(market)

    # Convert the given date to a pandas Timestamp object
    given_date = pd.Timestamp(given_date)

    # Get the schedule for the past month up to the given date
    one_month_ago = given_date - pd.DateOffset(months=1)
    schedule = market_cal.sessions_in_range(one_month_ago, given_date)

    # Find the last trading day before the given date
    last_trading_day = schedule[schedule < given_date][-1]

    return last_trading_day.to_pydatetime()


def market_settigs_date(cal_name: str, start: datetime, end: datetime):

    # Créer une instance du calendrier pour le marché spécifié
    calendar = ecals.get_calendar(cal_name)

    # Obtenir les jours ouvrés dans la plage de dates donnée
    trading_days = calendar.sessions_in_range(start, end)

    # Ajouter une heure après l'ouverture à chaque jour ouvré
    open_session = [calendar.session_open(session) for session in trading_days]

    # Ajouter une heure après l'ouverture à chaque jour ouvré
    settigs_hour = [(start_session - timedelta(hours=1),
                     start_session,
                     start_session + timedelta(hours=1))
                    for start_session in open_session]

    return settigs_hour


def Check_and_update_Date(tuples_list):
    now = datetime.now()
    # Utiliser une compréhension de liste pour filtrer les tuples dont la seconde date dépasse l'instant présent
    rebalDate = [(before, start, end) for before, start, end in tuples_list if end > now]

    # Vérifier si la liste est vide après le filtrage
    if not rebalDate:
        return False, rebalDate
    to_calib = rebalDate[0][0] <= now < rebalDate[0][1]
    # Vérifier si maintenant est entre le début et la fin du premier intervalle restant
    to_update = rebalDate[0][1] <= now < rebalDate[0][2]

    return to_calib, to_update, rebalDate
