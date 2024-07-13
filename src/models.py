import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import ListVector
from configs.rpy2_setup import setup_environment
from src.abstract import _Model
import src.common as cm
from src.data_mn import Data
from utils.load import load_json_config
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from utils.load import load_MLmodel, save_MLmodel
import numpy as np
import pandas as pd
from datetime import datetime
from configs.root_config import set_project_root

# Configure the path to the project root
set_project_root()


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
        # This configuration contains paths, model specifications, and other necessary configs.
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
            # Définir les packages à installer
            packages <- c("rugarch", "rmgarch")
            
            # Installer les packages si nécessaire
            for (pkg in packages) {
                if (!require(pkg, character.only = TRUE)) {
                    install.packages(pkg, repos = 'http://cran.rstudio.com/')
                    library(pkg, character.only = TRUE)
                }
            }

            
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
              
              # Create a multispecification model by replicating the univariate spec across 
              # the number of series in 'returns'.
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
        self.check_fit(data, horizon)
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

        mean = cm.weighting(means, scheme=self._model_config['model_config']["weights"])
        covariance = cm.weighting(covariances, scheme=self._model_config['model_config']["weights"])

        _, value = next(iter(data.data_config["cash"].items()))
        mean = np.insert(mean, 0, value)
        covariance = np.insert(np.insert(covariance, 0, 0, axis=1), 0, 0, axis=0)
        print(mean)
        print(covariance)
        metrics = {
            "fit_date": data.data_config["end_date"],
            "horizon": horizon,
            "scale": data.data_config["scale"],
            "mean": mean,
            "covariance": covariance,
            "to_update": True
        }
        # to take out of the if/else, if 2 model or plus
        self._metrics = metrics
        data.update_metrics(self)


class ML_Model(_Model):
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
        # This configuration contains paths, model specifications, and other necessary configs.
        super().__init__(model_config)

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

        # Data prep
        self.check_fit(data, horizon)
        model_available = os.path.exists(self._model_config["model_config"]["model_path"][0])
        brut_data = data._data.reset_index(drop=True)
        window_size = (horizon - data.data_config["end_date"]).days

        _returns, columns = cm.add_rolling_means(brut_data, window_size)
        shift_steps = -window_size
        returns = cm.shift_and_trim(_returns, columns, shift_steps)
        returns = cm.add_upper_triangle(returns, columns)

        y = returns.drop(data.data_config["symbols"], axis=1).values
        X = returns[data.data_config["symbols"]].values

        no_fit = model_available and not self.metrics["to_update"]
        model, scaler_X, scaler_y = None, None, None
        time_steps = 22
        # MODEL
        if no_fit:
            model, scaler_X, scaler_y = load_MLmodel(*self._model_config["model_config"]["model_path"])

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)
            X_train, y_train = cm.create_sequences(X_scaled, y_scaled, time_steps)

            model.compile(optimizer='adam', loss='mean_squared_error')
            # ajustement du modèle avec les nouvelles données
            model.fit(X_train, y_train,
                      epochs=1, batch_size=20,
                      validation_split=0.2,
                      verbose=1)

        if not (model and scaler_X and scaler_y):
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)
            X_train, y_train = cm.create_sequences(X_scaled, y_scaled, time_steps)
            # Defining the LSTM model with an Input layer
            def create_model():
                model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    Conv1D(filters=4, kernel_size=2, activation='relu', padding='same'),
                    Conv1D(filters=4, kernel_size=2, activation='relu', padding='same'),
                    MaxPooling1D(pool_size=1),
                    Dropout(0.2),
                    Flatten(),
                    Dense(16, activation='relu'),  # Couche intermédiaire ajoutée
                    Dense(8, activation='relu'),
                    Dense(y_train.shape[1])
                ])

                model.compile(optimizer=Nadam(learning_rate=0.001), loss='mean_squared_error')

                return model

            num_models = 10  # Nombre de modèles à entraîner
            models = []
            histories = []
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            for i in range(num_models):
                model = create_model()
                print(f"Training model {i + 1}/{num_models}")
                history = model.fit(X_train, y_train,
                                    epochs=300, batch_size=32,
                                    validation_split=0.2,
                                    callbacks=[early_stopping],
                                    verbose=1)
                models.append(model)
                histories.append(history)

            # Sélection du meilleur modèle basé sur la performance de validation
            val_losses = [history.history['val_loss'][-1] for history in histories]
            best_model_index = np.argmin(val_losses)
            model = models[best_model_index]

            #save_MLmodel(model, scaler_X, scaler_y,
                    #*self._model_config["model_config"]["model_path"])

        new_X = cm.sequence_for_predict(data.data[data.data_config["symbols"]].values, time_steps)

        new_X_scaled = scaler_X.transform(new_X)

        # Ajouter une dimension pour correspondre à l'entrée attendue par le modèle LSTM (échantillon, time_steps,
        # features)
        new_X_scaled = np.expand_dims(new_X_scaled, axis=0)

        # Faire la prédiction avec le modèle
        new_y_scaled = model.predict(new_X_scaled)

        # Inverser la transformation pour revenir à l'échelle d'origine
        new_y = scaler_y.inverse_transform(new_y_scaled).flatten()

        nb_asset = len(data.data_config["symbols"])
        mean = np.array(new_y[:nb_asset])
        mean_matrix = mean.reshape(-1, 1)
        covariance = np.array(cm.reconstruct_matrix(new_y[nb_asset:])) - np.dot(mean_matrix, mean_matrix.T)
        _, value = next(iter(data.data_config["cash"].items()))
        mean = np.insert(mean, 0, value)
        covariance = np.insert(np.insert(covariance, 0, 0, axis=1), 0, 0, axis=0)
        print(mean)
        print(covariance)
        metrics = {
            "fit_date": data.data_config["end_date"],
            "horizon": horizon,
            "scale": data.data_config["scale"],
            "mean": mean,
            "covariance": covariance,
            "to_update": True
        }

        # to take out of the if/else, if 2 model or plus
        self._metrics = metrics
        data.update_metrics(self)

"""
data_config = load_json_config(r'src/data_settings/data_settings.json')
model_config = load_json_config(r'src/model_settings/model_settings_LSTM.json')
model_config2 = load_json_config(r'src/model_settings/model_settings.json')
data = Data(data_config)
data.fetch_data(pd.Timestamp(year=2005, month=1, day=1).tz_localize('UTC').tz_localize(None),
                pd.Timestamp(year=2018, month=1, day=3).tz_localize('UTC').tz_localize(None))
model = Lstm_Model(model_config)
model2 = Model(model_config2)
horizon = pd.Timestamp(year=2018, month=1, day=10).tz_localize('UTC').tz_localize(None)
model.fit_fcast(data, horizon)
model2.fit_fcast(data, horizon)
"""
