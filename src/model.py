import os
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from src.data_monitoring import StockData
from src.rpy2_setup import setup_environment
import utils.load_model as lo_m

class DCCGARCHModel:
    def __init__(self, data: StockData, model_info: str):
        """
        Initialize the DCC GARCH Model with necessary data and configurations.

        Parameters:
        data (StockData): The stock data manager containing market data.
        model_info (str): Path to the model configuration file.

        This method sets up the class by storing the stock data and model info,
        initializing the forecast to None, setting up the R environment,
        and defining necessary R functions for DCC GARCH analysis.
        """
        self.data = data
        self.model_info = model_info
        self.forecast = None
        self.setup_environment()
        self.define_r_functions()

    def setup_environment(self):
        """
        Set up the R environment by loading necessary libraries.
        This method assumes that 'setup_environment' from 'src.rpy2_setup' properly
        configures the R environment, including loading any required R packages.
        """
        setup_environment()

    def define_r_functions(self):
        """
        Define R functions necessary for DCC GARCH model analysis.
        This includes the R function for executing the model fitting and forecasting.

        It loads the 'rmgarch' package necessary for running multivariate GARCH models
        and defines a function 'run_dcc_garch_and_forecast' in R that handles:
        - Loading or fitting a DCC GARCH model.
        - Forecasting based on the fitted model.
        - Handling errors in reading or saving models.
        - Returning forecast results including means and covariances.
        """
        ro.r('''
        library(rmgarch)
        run_dcc_garch_and_forecast <- function(returns, model_path, model_available, n_ahead) {
            dccfit <- NULL
            if (model_available) {
                dccfit <- tryCatch({ readRDS(model_path) }, error=function(e) {
                    print(paste("Error reading RDS:", e$message))
                    NULL
                })
            }
            if (is.null(dccfit)) {
                spec <- ugarchspec(variance.model=list(model="sGARCH"), mean.model=list(armaOrder=c(2, 1)), distribution.model="norm")
                multispec <- multispec(replicate(ncol(returns), spec))
                dccspec <- dccspec(uspec=multispec, dccOrder=c(1, 1), distribution="mvt")
                dccfit <- dccfit(dccspec, data=returns, out.sample=0)
                tryCatch({ saveRDS(dccfit, file=model_path) }, error=function(e) {
                    print(paste("Failed to save model:", e$message))
                })
            }
            fcast <- dccforecast(dccfit, n.ahead=n_ahead)
            covariance <- lapply(1:n_ahead, function(i) fcast@mforecast$H[[1]][,,i])
            return(list(means=fcast@mforecast$mu, covariances=covariance))
        }
        ''')

    def _forecast(self, n_ahead: int = 5):
        """
        Perform forecasting using the defined DCC GARCH model.

        Parameters:
        n_ahead (int): The number of periods ahead for which to forecast.

        Returns:
        dict: A dictionary containing the forecast results, including means and covariances.

        This method activates the interface between pandas and R, converts stock data to an R-compatible format,
        checks model availability, and executes the R forecasting function. The results are stored and returned.
        """
        config = lo_m.load_json_config(self.model_info)
        pandas2ri.activate()
        r_returns = pandas2ri.py2rpy(self.data.data)
        model_available = set(config["symbols"]) == set(self.data.symbols)
        results = ro.globalenv['run_dcc_garch_and_forecast'](r_returns, config["model"], model_available, n_ahead)
        self.forecast = {"means": [np.array(vec).flatten() for vec in results.rx2('means')],
                         "covariances": [np.array(vec) for vec in results.rx2('covariances')]}
        return self.forecast

# Usage example
symbols = ['AAPL', 'MSFT', 'GOOGL']
stock_data_manager = StockData(symbols)
stock_data_manager.fetch_data('2018-01-01', '2020-01-10')
print(stock_data_manager.data)  # Initial data

model_info = 'C:/Users/MatarKANDJI/automAM/src/settings/stocks_settings.json'
dcc_garch_model = DCCGARCHModel(stock_data_manager, model_info)
forecast_results = dcc_garch_model._forecast()
print(forecast_results)
