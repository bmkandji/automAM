import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from src.common import compute_log_returns
from src.abstract import _Data

class Data(_Data):
    def __init__(self, data_config: dict):
        """
        Initialize the Data object with stock config.

        Parameters:
        data_config (str): Path to the data configuration file.
        """
        super().__init__(data_config)


    def fetch_data(self, start_date: datetime, end_date: datetime) -> int:
        """
        Fetches the daily closing prices for a list of stock symbols over a specified date range, calculates log returns,
        applies a rolling mean to handle NA values, and appends the data to the internal DataFrame. This method ensures
         that the end date specified is inclusive by adjusting the date range internally.

        Parameters:
        start_date (str): The starting date of the period (format: 'YYYY-MM-DD').
        end_date (str): The ending date of the period (format: 'YYYY-MM-DD'), adjusted to be inclusive.

        Returns:
        int: The number of new rows added to the DataFrame after processing.
        """
        new_data = pd.DataFrame()
        for symbol in self.data_config["symbols"]:
            try:
                # Fetch stock data from Yahoo Finance
                stock_data = yf.download(symbol, start=start_date, end=end_date)
                # Create a temporary DataFrame to store the close prices
                temp_data = pd.DataFrame({
                    'Date': stock_data.index,
                    'Close': stock_data['Adj Close'],  # Use 'Adj Close' for adjusted closing prices
                    'Symbol': symbol
                })
                new_data = pd.concat([new_data, temp_data])
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        # Remove duplicates and reset index
        if not new_data.empty:
            new_data.set_index('Date', inplace=True)
            new_data = compute_log_returns(new_data.pivot(columns='Symbol', values='Close'), self.data_config["scale"])
            initial_data_length = len(self.data)
            self._data = pd.concat([self.data, new_data]).drop_duplicates()
            self.replace_NA()
            self._data_config["end_date"] = self.data.index.max()
            new_data_length = len(self.data) - initial_data_length
            return new_data_length

        return 0

    def update_data(self, new_end_date: datetime = None):
        """
        Updates the internal DataFrame by fetching new data from the day after the last recorded end date to a new end
        date, and removes an equal number of old data rows from the start.

        Parameters:
        new_end_date (str, optional): The end date for the new data fetch (format: 'YYYY-MM-DD'). Defaults to today's
        date.
        """
        if new_end_date is None:
            new_end_date = datetime.today().strftime('%Y-%m-%d')
        start_date_update = self.data.index.max()
        new_rows_added = self.fetch_data(start_date_update, new_end_date)

        # Remove the same number of oldest rows as new rows added
        if len(self.data) > new_rows_added:
            self._data = self.data.iloc[new_rows_added:]  # Keeps the DataFrame size consistent
            self._data_config["start_date"] = self.data.index.min()

    def window_returns(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """
        Filter the instance's DataFrame based on a date range, exclusive of the start date and inclusive of the end date,
        and return an array containing the sum of each column in the filtered DataFrame.

        Args:
        - start_date (datetime): The start date, exclusive.
        - end_date (datetime): The end date, inclusive.

        Returns:
        - np.ndarray: An array containing the sum of each column from the filtered DataFrame.

        Raises:
        - ValueError: If no data is available for the given date range.
        """
        # Ensure the index is in datetime format and filter the DataFrame
        self._data.index = pd.to_datetime(self.data.index)
        filtered_df = self.data.loc[(self.data.index > start_date) & (self.data.index <= end_date)]

        # Check if the filtered DataFrame is empty
        if filtered_df.empty:
            raise ValueError(f"No data available from {start_date} to {end_date}.")

        return filtered_df.sum()

    def replace_NA(self, window: int = 5) -> None:
        """
        Replaces NA values in the DataFrame with the rolling mean calculated over a specified window size.

        Parameters:
        window (int): The size of the rolling window to calculate the means, default is 5.
        """
        # Calculate the rolling mean with a specified window, minimum number of observations in the window required
        # to have a value is 1
        roll_means = self.data.rolling(window=window, min_periods=1, center=True).mean()

        # Replace NA values in the DataFrame with the calculated rolling means
        self.data.fillna(roll_means, inplace=True)
