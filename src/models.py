import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import ListVector
from src.data_mn import Data
from settings.rpy2_setup import setup_environment
from src.common import weighting
from datetime import datetime
from src.abstract import _Model


class Model(_Model):
    def __init__(self, model_config: dict):
        """
        Initialize the DCC GARCH Model with necessary data and configurations.

        Parameters:
        data (Data): The stock data manager containing market data.
        model_config (str): Path to the model configuration file.

        This method sets up the class by storing the stock data and model info,
        initializing the forecast to None, setting up the R environment,
        and defining necessary R functions for DCC GARCH analysis.
        """

        # Load the JSON configuration for the model using a utility function.
        # This configuration contains paths, model specifications, and other necessary settings.
        super().__init__(model_config)
        self.stp_environment()
        self.define_r_functions()

    @staticmethod
    def stp_environment():
        """
        Set up the R environment by loading necessary libraries.
        This method assumes that 'setup_environment' from 'src.rpy2_setup' properly
        configures the R environment, including loading any required R packages.
        """
        setup_environment()

    @staticmethod
    def define_r_functions():
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
            # Load the 'rmgarch' package, necessary for running multivariate GARCH models.
            library(rmgarch)
            
            # Define a function to run and forecast a DCC GARCH model based on specified parameters.
            run_dcc_garch_and_forecast <- function(returns, model_config, n_ahead, no_fit) {
              # Initialize variables to store the coefficients and the fitted model.
              coef <- NULL
              dccfit <- NULL
            
              # Define the GARCH model specification using parameters from 'model_config'.
              spec <- ugarchspec(
                variance.model = list(model = model_config$model), 
                mean.model = list(armaOrder = model_config$armaOrder), 
                distribution.model = model_config$distribution_garch
              )
              
              # Create a multispecification model by replicating the univariate spec across the number of series in 'returns'.
              multispec <- multispec(replicate(ncol(returns), spec))           
              
              # Attempt to load existing model coefficients if no fitting is indicated.
              if (no_fit) {
                coef <- tryCatch({
                  readRDS(model_config$model_path)  # Read the model coefficients from the specified path.
                }, error = function(e) {
                  print(paste("Error reading RDS:", e$message))  # Log error message if reading fails.
                  NULL  # Return NULL if an error occurs.
                })
              }
              
              # Define the DCC model specification, incorporating loaded coefficients if available.
              dccspec <- dccspec(
                uspec = multispec, 
                dccOrder = model_config$dccOrder, 
                distribution = model_config$distribution_dcc,
                fixed.pars = coef  # This will be NULL if 'coef' is not loaded, handled by 'dccspec'.
              )
              
              # Fit the DCC model using the specified data.
              dccfit <- dccfit(dccspec, data = returns, out.sample = 0)
            
              # If no coefficients were loaded, save the newly fitted model's coefficients.
              if (is.null(coef)) {
                coef <- coef(dccfit)  # Extract coefficients from the fitted model.
                tryCatch({
                  saveRDS(coef, file = model_config$model_path)  # Save the coefficients to the specified path.
                }, error = function(e) {
                  print(paste("Failed to save model:", e$message))  # Log error if saving fails.
                })
              }
              
              # Forecast using the fitted model for 'n_ahead' periods.
              fcast <- dccforecast(dccfit, n.ahead = n_ahead)
              
              # Calculate covariance matrices for each forecast period.
              covariance <- lapply(1:n_ahead, function(i) fcast@mforecast$H[[1]][,,i])
              
              # Return a list containing the mean forecasts and covariance matrices.
              return(list(means = fcast@mforecast$mu, covariances = covariance))
            }
            ''')

    def fit_fcast(self, data: Data, horizon: datetime):

        """
        Perform forecasting using the defined DCC GARCH model.

        Parameters:
        n_ahead (int): The number of periods ahead for which to forecast.

        Returns:
        dict: A dictionary containing the forecast results, including means and covariances.

        This method activates the interface between pandas and R, converts stock data to an R-compatible format,
        checks model availability, and executes the R forecasting function. The results are stored and returned.
        """

        # Activate the automatic conversion of pandas data structures to R data structures.
        # This is crucial for passing pandas DataFrame or Series objects directly to R functions.
        pandas2ri.activate()

        # Convert the DataFrame stored in 'data.data' to an R data.frame using rpy2's conversion.
        # This is necessary because R functions expect data in R data.frame format.
        # noinspection PyProtectedMember
        r_returns = pandas2ri.py2rpy(data._data)

        # Check if the symbols in the configuration match those in the data.
        # This is a form of validation to ensure that the data being processed is as expected.
        model_available = os.path.exists(self._model_config["model_config"]["model_path"])

        # Create an R list vector to hold the configuration parameters for the R function.
        # Each parameter is converted to the appropriate R type, such as using IntVector for integer arrays.
        model_config_vector = ListVector({
            'model_path': self.model_config['model_config']['model_path'],  # Path to the model file.
            'model': self.model_config['model_config']['model'],  # Model type, e.g., 'sGARCH'.
            'armaOrder': ro.IntVector(self.model_config['model_config']['armaOrder']),
            # ARMA order as an integer vector.
            'dccOrder': ro.IntVector(self.model_config['model_config']['dccOrder']),  # DCC model order.
            'distribution_garch': self.model_config['model_config']['distribution_garch'],  # GARCH distribution.
            'distribution_dcc': self.model_config['model_config']['distribution_dcc']  # DCC distribution.
        })
        n_ahead = (horizon - data.data_config["end_date"]).days
        # Call the R function 'run_dcc_garch_and_forecast' with the necessary parameters.
        # This function is expected to perform GARCH modeling and forecasting.

        no_fit = model_available and not self.metrics["to_update"]
        results = ro.globalenv['run_dcc_garch_and_forecast'](r_returns, model_config_vector, n_ahead, no_fit)

        # Process the returned results from R, extracting means and covariances.
        # Convert them to numpy arrays for easier manipulation and use in Python.
        means = np.array([np.array(vec).flatten() for vec in results.rx2('means')])
        covariances = np.array([np.array(vec) for vec in results.rx2('covariances')])

        mean = weighting(means, scheme=self._model_config['model_config']["weights"])
        covariance = weighting(covariances, scheme=self._model_config['model_config']["weights"])

        _, value = next(iter(data.data_config["currency"].items()))
        mean = np.insert(mean, 0, value)
        covariance = np.insert(np.insert(covariance, 0, 0, axis=1), 0, 0, axis=0)

        metrics = {
            "fit_date": data.data_config["end_date"],
            "scale": data.data_config["scale"],
            "mean": mean,
            "covariance": covariance
        }
        # to take out of the if/else, if 2 model or plus
        self._metrics = metrics
        data.update_metrics(self)
