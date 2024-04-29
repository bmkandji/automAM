import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class StockData:
    def __init__(self, symbols: list[str]):
        """
        Initialize the StockData object with stock symbols.

        Parameters:
        symbols (list[str]): A list of stock symbols for fetching data.
        """
        self.symbols = symbols
        self.end_date = None
        self.data = pd.DataFrame()

    def fetch_data(self, start_date: str, end_date: str) -> int:
        """
        Fetches the daily closing prices for a list of stock symbols over a specified date range and appends
        the data to the internal DataFrame.

        Parameters:
        start_date (str): The starting date of the period (format: 'YYYY-MM-DD').
        end_date (str): The ending date of the period (format: 'YYYY-MM-DD').

        Returns:
        int: The number of new rows added to the DataFrame.
        """
        new_data = pd.DataFrame()
        for symbol in self.symbols:
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
        # Update the end date in the object
        self.end_date = end_date
        # Remove duplicates and reset index
        if not new_data.empty:
            new_data.set_index('Date', inplace=True)
            new_data = new_data.pivot(columns='Symbol', values='Close')
            initial_data_length = len(self.data)
            self.data = pd.concat([self.data, new_data]).drop_duplicates()
            new_data_length = len(self.data) - initial_data_length
            return new_data_length

        return 0

    def update_data(self, new_end_date: str = None):
        """
        Updates the internal DataFrame by fetching new data from the day after the last recorded end date to a new end date,
        and removes an equal number of old data rows from the start.

        Parameters:
        new_end_date (str, optional): The end date for the new data fetch (format: 'YYYY-MM-DD'). Defaults to today's date.
        """
        if new_end_date is None:
            new_end_date = datetime.today().strftime('%Y-%m-%d')
        start_date_update = (datetime.strptime(self.end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        new_rows_added = self.fetch_data(start_date_update, new_end_date)

        # Remove the same number of oldest rows as new rows added
        if len(self.data) > new_rows_added:
            self.data = self.data.iloc[new_rows_added:]  # Keeps the DataFrame size consistent


# Example usage
symbols = ['AAPL', 'MSFT', 'GOOGL']
stock_data_manager = StockData(symbols)
stock_data_manager.fetch_data('2020-01-01', '2020-12-31')
print(stock_data_manager.data)  # Print the initial data

# Update this data with new entries and manage the size
stock_data_manager.update_data()
print(stock_data_manager.data)  # Print the updated data
