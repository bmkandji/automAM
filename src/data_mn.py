import yfinance as yf
from src.common import compute_log_returns
from src.abstract import _Data
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta


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
                stock_data = yf.download(symbol, start=start_date, end=end_date-timedelta(hours=12))
                # Create a temporary DataFrame to store the close prices
                temp_data = pd.DataFrame({
                    'Date': stock_data.index.tz_localize('UTC').tz_localize(None),
                    'Close': stock_data['Adj Close'],  # Use 'Adj Close' for adjusted closing prices
                    'Symbol': symbol
                })
                new_data = pd.concat([new_data, temp_data])
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        # Remove duplicates and reset index
        if not new_data.empty:
            new_data.set_index('Date', inplace=True)
            new_data = compute_log_returns(
                new_data.pivot(columns='Symbol', values='Close')[self.data_config["symbols"]],
                self._data_config["scale"])
            initial_data_length = len(self._data)
            self._data = pd.concat([self._data, new_data]).drop_duplicates()
            self.replace_NA()
            self._data_config["end_date"] = self._data.index.max()
            new_data_length = len(self._data) - initial_data_length
            return new_data_length
        self._data_config["start_date"] = self._data.index.min()
        return 0


class AlpacaData(_Data):
    def __init__(self, data_config: dict):
        """
        Initialize the Data object with stock config.

        Parameters:
        data_config (str): Path to the data configuration file.
        """
        super().__init__(data_config)
        self.api = tradeapi.REST(
            data_config["api_config"]['api_key'],
            data_config["api_config"]['api_secret'],
            data_config["api_config"]['base_url'],
            api_version='v2'
        )

    def fetch_data(self, start_date: datetime, end_date: datetime) -> int:
        """
        Fetches the daily closing prices for a list of stock symbols over a specified date range using Alpaca API,
        calculates log returns, applies a rolling mean to handle NA values, and appends the data to the internal DataFrame.
        Adjusts the date range internally to include the end date.

        Parameters:
        start_date (datetime): The starting date of the period.
        end_date (datetime): The ending date of the period, adjusted to be inclusive.

        Returns:
        int: The number of new rows added to the DataFrame after processing.
        """
        new_data = pd.DataFrame()

        # Adjust end_date to be inclusive and format dates
        formatted_start_date = (start_date - timedelta(days=1)).strftime('%Y-%m-%d')
        formatted_end_date = end_date.strftime('%Y-%m-%d')

        for symbol in self.data_config["symbols"]:
            try:
                # Fetch stock data from Alpaca API using get_bars
                bars = self.api.get_bars(symbol, tradeapi.TimeFrame.Day,
                                         formatted_start_date, formatted_end_date,
                                         adjustment='raw').df

                # Create a temporary DataFrame to store the close prices
                if not bars.empty:
                    temp_data = pd.DataFrame({
                        'Date': bars.index.date,
                        'Close': bars['close'],
                        'Symbol': symbol
                    })
                    new_data = pd.concat([new_data, temp_data])
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        if not new_data.empty:
            new_data.set_index('Date', inplace=True)
            new_data = compute_log_returns(
                new_data.pivot(columns='Symbol', values='Close')[self.data_config["symbols"]],
                self._data_config["scale"])
            initial_data_length = len(self._data)
            new_data.index = pd.to_datetime(new_data.index).tz_localize('UTC').tz_localize(None)
            self._data = pd.concat([self._data, new_data]).drop_duplicates()
            self.replace_NA()
            self._data_config["end_date"] = self._data.index.max()
            new_data_length = len(self._data) - initial_data_length
            return new_data_length
        self._data_config["start_date"] = self._data.index.min()
        return 0
