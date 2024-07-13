import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Dict
from decimal import Decimal, getcontext, ROUND_DOWN
import exchange_calendars as ecals
from datetime import datetime, timedelta
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K


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


def get_last_trading_day(given_date: pd.Timestamp, market: str) -> datetime:
    """
    Returns the last trading day for the specified market before the given date.

    :param given_date: The given date as a string in the format '2024-05-17 15:59:52.543394924-04:00'.
    :param market: The market identifier, e.g., 'XNYS' for New York Stock Exchange.
    :return: The last trading day as a datetime object.
    """
    # Convert to UTC and then make it timezone naive
    given_date = datetime(given_date.year, given_date.month, given_date.day)

    # Get the calendar for the specified market
    market_cal = ecals.get_calendar(market)
    # Get the schedule for the past month up to the given date
    one_month_ago = given_date - pd.DateOffset(months=1)
    schedule = market_cal.sessions_in_range(one_month_ago, given_date)

    # Check if there are trading days in the schedule
    if len(schedule) == 0:
        raise ValueError("No trading days found in the past month up to the given date.")

    # Find the last trading day before the given date
    last_trading_day = schedule[schedule < given_date][-1]

    # Convert the last trading day to UTC
    last_trading_day = pd.Timestamp(last_trading_day).tz_localize('UTC').tz_localize(None)

    return last_trading_day


def market_settings_date(cal_name: str, start: datetime, end: datetime) -> List[Tuple[datetime, datetime, datetime]]:
    """
    Generates a list of tuples containing the market settings dates with open, close, and extended hours.

    :param cal_name: The name of the market calendar.
    :param start: The start date as a datetime object.
    :param end: The end date as a datetime object.
    :return: A list of tuples with market open, close, and extended hours dates.
    """
    # Create a calendar instance for the specified market
    calendar = ecals.get_calendar(cal_name)

    # Ensure start and end dates are datetime objects
    start = datetime(start.year, start.month, start.day)
    end = datetime(end.year, end.month, end.day)

    # Get the last session available in the calendar
    last_session = calendar.last_session

    # Initialize the list of open sessions
    open_sessions = []

    # Get trading days in the given date range up to the last session available
    trading_days = calendar.sessions_in_range(start, min(end, last_session))
    open_sessions.extend([calendar.session_open(session) for session in trading_days])

    # If the end date exceeds the last session, generate future dates manually
    current_date = last_session + timedelta(days=1)
    while current_date <= end:
        if current_date.weekday() < 5:  # Monday to Friday
            # Only add the session if the market has defined an open time
            if current_date in calendar.opens.index:
                open_sessions.append(calendar.opens.loc[current_date])
        current_date += timedelta(days=1)

    # Add one hour before and after the open time to each trading day
    settings_hours = [(open_session - timedelta(hours=10),
                       open_session,
                       open_session + timedelta(hours=10))
                      for open_session in open_sessions]

    return settings_hours


def Check_and_update_Date(tuples_list) -> list:
    now = get_current_time()
    # Utiliser une compréhension de liste pour filtrer les tuples dont la seconde date dépasse l'instant présent
    rebalDate = [(before, start, end) for before, start, end in tuples_list if end > now]

    # Vérifier si la liste est vide après le filtrage
    if not rebalDate:
        return False, False, rebalDate
    to_calib = rebalDate[0][0] <= now < rebalDate[0][1]
    # Vérifier si maintenant est entre le début et la fin du premier intervalle restant
    to_update = rebalDate[0][1] <= now < rebalDate[0][2]

    return to_calib, to_update, rebalDate


def trunc_decimal(number: float, decimals: int) -> float:
    """
    Truncate a decimal number to a specific number of decimal places.

    :param number: The decimal value to be truncated.
    :param decimals: The number of decimal places to truncate to.
    :return: The truncated decimal value.
    """
    # Définir le mode d'arrondi à ROUND_DOWN pour tronquer
    getcontext().rounding = ROUND_DOWN
    # Créer le facteur de quantification basé sur le nombre de décimales souhaité
    factor = Decimal('1.' + '0' * decimals)
    # Quantizer le nombre à ce facteur
    return float(Decimal(number).quantize(factor))


def normalize_order(orders: List[Dict], min_value: float = 1.0) -> List[Dict]:
    """
    Filters a list of orders to only include those with a value greater than a specified minimum value.

    Parameters:
    orders (List[Dict]): A list of dictionaries, each representing an order with at least a 'value' key.
    min_value (float): The minimum value threshold for orders to be included in the result. Default is 1.0.

    Returns:
    List[Dict]: A list of dictionaries containing only the orders with a 'value' greater than min_value.
    """
    # Use a list comprehension to filter the orders
    return [
        order  # Include the order in the resulting list
        for order in orders  # Iterate through each order in the input list
        if order["value"] > min_value  # Only include orders where the 'value' key is greater than min_value
    ]

def get_current_time():
    # URL de l'API World Time pour UTC
    url = "http://worldtimeapi.org/api/timezone/Etc/UTC"

    # Faire une requête GET à l'API
    response = requests.get(url)

    if response.status_code == 200:
        # Extraire les données JSON de la réponse
        data = response.json()

        # Obtenir l'heure actuelle en UTC depuis les données JSON
        utc_time_str = data['datetime']

        # Convertir la chaîne de caractères en objet datetime avec fuseau horaire
        utc_time = datetime.fromisoformat(utc_time_str.replace('Z', '+00:00'))

        return utc_time
    else:
        # Gestion des erreurs
        print(f"Erreur lors de la récupération de l'heure UTC : {response.status_code}")
        return None


####################################################################################
####################################################################################
################################# LSTM tools #######################################
####################################################################################
####################################################################################


def add_rolling_means(returns, window):
    column_names = []
    for column in returns.columns:
        rolling_mean_col = returns[column].rolling(window=window).mean()
        column_name = f"{column}_rolling_mean_{window}"
        returns[column_name] = rolling_mean_col
        column_names.append(column_name)
    return returns, column_names


def shift_and_trim(returns, columns, shift_steps):
    # Décaler les colonnes spécifiées
    returns[columns] = returns[columns].shift(shift_steps)

    # Supprimer les lignes avec des valeurs manquantes
    returns = returns.dropna()

    return returns


def add_upper_triangle(rends, columns):
    returns = rends.copy()
    n = len(columns)
    for i in range(n):
        for j in range(i, n):
            col_name = f'{columns[i]}_x_{columns[j]}'
            # Explicitly use .loc to avoid SettingWithCopyWarning
            returns.loc[:, col_name] = returns.loc[:, columns[i]] * returns.loc[:, columns[j]]
    return returns


def reconstruct_matrix(vector):
    # Calculer la taille de la matrice
    n = int((-1 + (1 + 8 * len(vector))**0.5) / 2)
    matrix = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = vector[index]
            if i != j:
                matrix[j, i] = vector[index]
            index += 1
    return matrix


def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[(i+1):(i + time_steps + 1)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def sequence_for_predict(X, time_steps=1):
    result = X[(len(X) - time_steps):len(X)]
    return result
