import os
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from src.data_monitoring import StockData
from src.rpy2_setup import setup_environment
import utils.load_model as lo_m


def define_r_functions():
    """ Define R functions for DCC GARCH model estimation and forecasting.
        This function loads necessary R packages and defines the R function
        for running DCC GARCH analysis, handling model persistence, and forecasting.
    """
    setup_environment()  # Setup R environment, loading libraries if needed.
    ro.r('''
    library(rmgarch)  # Load the rmgarch package for multivariate GARCH models.

    # Define the R function for executing the DCC-GARCH model fitting and forecasting.
    run_dcc_garch_and_forecast <- function(returns, model_path, model_available, n_ahead) {
        dccfit <- NULL  # Initialize dccfit object to NULL.

        # Attempt to load an existing model if it is available.
        if (model_available) {
            dccfit <- tryCatch({
                readRDS(model_path)
            }, error = function(e) {
                print(paste("Error reading RDS:", e$message))
                NULL  # Return NULL if there is an error reading the model.
            })
        }

        # If no model is loaded, fit a new model.
        if (is.null(dccfit)) {
            print("No existing model found, creating a new one.")
            spec <- ugarchspec(variance.model = list(model = "sGARCH"),
                               mean.model = list(armaOrder = c(2, 1)),
                               distribution.model = "norm")
            multispec <- multispec(replicate(ncol(returns), spec))
            dccspec <- dccspec(uspec = multispec, dccOrder = c(1, 1), distribution = "mvt")
            dccfit <- dccfit(dccspec, data = returns, out.sample = 0)

            # Try saving the newly created model to the specified path.
            tryCatch({
                saveRDS(dccfit, file = model_path)
                print(paste("Model successfully saved to:", model_path))
            }, error = function(e) {
                print(paste("Failed to save model:", e$message))
            })
        }

        # Forecast using the fitted model.
        fcast <- dccforecast(dccfit, n.ahead = n_ahead)
        return(list( means = fcast@mforecast$mu,
                    variances = fcast@mforecast$H))
    }
    ''')


def mean_variance_forecast(stock_data_manager, model_info: str = None, n_ahead : int =21):
    """ Perform mean and variance forecasting using DCC GARCH model.

    Args:
        stock_data_manager (StockData): The stock data manager containing market data.
        model_info (str): Path to the model configuration file.
        n_ahead (int): Number of days ahead for the forecast.

    Returns:
        dict: A dictionary containing the model fit, mean forecasts, and variance forecasts.
    """
    # Load configuration for the model.
    config = lo_m.load_json_config(model_info)
    pandas2ri.activate()  # Activate conversion between pandas and R data frames.

    # Convert stock data to R-compatible format.
    r_returns = pandas2ri.py2rpy(stock_data_manager.data)

    # Check if the existing model's symbols match the current stock data symbols.
    model_available = set(config["symbols"]) == set(stock_data_manager.symbols)

    # Define R functions needed for forecasting.
    define_r_functions()

    # Run the forecasting function defined in R and return results.
    results = ro.globalenv['run_dcc_garch_and_forecast'](r_returns, config["model"], model_available, n_ahead)

    means = [vec for vec in results.rx2('means')]
    variances = [vec for vec in results.rx2('variances')]
    return {"means": results.rx2('means'), "covariances": results.rx2('variances')}


# Example usage
symbols = ['AAPL', 'MSFT', 'GOOGL']
stock_data_manager = StockData(symbols)
stock_data_manager.fetch_data('2018-01-01', '2020-01-10')
print(stock_data_manager.data)  # Initial data

stock_data_manager.update_data('2020-01-15')
print(stock_data_manager.data)  # Updated data

# Forecast
model_info = 'C:/Users/MatarKANDJI/automAM/src/settings/stocks_settings.json'
results = mean_variance_forecast(stock_data_manager, model_info=model_info, n_ahead=10)

means = results['means']
variances = results['covariances']


print(means)
print(variances)
