from typing import Any
import numpy as np
from datetime import datetime
from typing import Optional
import numpy as np
from numpy import ndarray, dtype, floating
from scipy.optimize import minimize


def check_inputs(asset_weights, asset_expected_returns, initial_capital):
    """
    Check that the inputs to the portfolio return and variance calculations are valid.

    Parameters: asset_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. asset_expected_returns (numpy.ndarray): Array of asset expected
    returns. Each element represents the expected return of the corresponding asset. initial_capital (float,
    optional): The initial capital invested in the portfolio. Defaults to 1.

    Raises:
    ValueError: If the number of assets in asset_weights does not match the number of assets in asset_expected_returns,
                or if initial_capital is not a positive number.
    """
    # Check that asset_weights and asset_expected_returns have the same length
    if len(asset_weights) != len(asset_expected_returns):
        raise ValueError("The number of assets in asset_weights must match the number of assets in "
                         "asset_expected_returns.")

    # Check that initial_capital is positive
    if initial_capital < 0:
        raise ValueError("initial_capital must be a positive number.")


def portfolio_return(asset_weights: np.ndarray, expected_returns: np.ndarray, initial_capital: float = 1) -> ndarray[
    Any, dtype[floating[Any]]]:
    """
    Calculate the expected return of the portfolio.

    Parameters: asset_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. expected_returns (numpy.ndarray): Array of asset expected returns.
    Each element represents the expected return of the corresponding asset. initial_capital (float, optional): The
    initial capital invested in the portfolio. Defaults to 1.

    Returns: float: Expected return of the portfolio. This is calculated as the dot product of the weights array and
    the expected returns array, scaled by the initial capital.

    This function calculates the portfolio return by taking the dot product of the asset weights and the asset
    expected returns. The return is then scaled by the initial capital.
    """
    check_inputs(asset_weights, expected_returns, initial_capital)

    # Calculate and return the portfolio return
    return initial_capital * (asset_weights.T @ expected_returns)


def portfolio_variance(asset_weights: np.ndarray, covariance_matrix: np.ndarray, initial_capital: float = 1) -> ndarray[
    Any, dtype[floating[Any]]]:
    """
    Calculate the expected variance of the portfolio.

    Parameters: asset_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. covariance_matrix (numpy.ndarray): Expected covariance matrix.
    This is a 2D array where the element at the i-th row and j-th column represents the covariance between the i-th
    and j-th assets. initial_capital (float, optional): The initial capital invested in the portfolio. Defaults to 1.

    Returns: float: Expected variance of the portfolio. This is calculated as the dot product of the weights array
    and the product of the weights array and the covariance matrix.

    This function calculates the portfolio variance by taking the dot product of the asset weights and the product of
    the asset weights and the covariance matrix. The variance is scaled by the square of the initial capital.
    """
    check_inputs(asset_weights, covariance_matrix, initial_capital)

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be a square matrix.")

    # Calculate and return the portfolio variance
    return initial_capital ** 2 * (asset_weights.T @ covariance_matrix @ asset_weights)


import cvxpy as cp


def transaction_costs(weights, current_weights, transaction_fee_rate, initial_capital: float = 1):
    """
    Calculate the transaction costs for rebalancing the portfolio.

    Parameters:
    - weights (cvxpy.Variable): The variable representing the new weights of the assets in the portfolio.
    - current_weights (np.ndarray): Current portfolio weights before rebalancing.
    - transaction_fee_rate (float): The transaction fee rate as a percentage of the traded amount.

    Returns:
    - cvxpy.Expression: The total transaction costs incurred when rebalancing.
    """
    return initial_capital * transaction_fee_rate * cp.sum(cp.abs(weights - current_weights))


def portfolio_volatility(asset_weights: np.ndarray, covariance_matrix: np.ndarray, initial_capital: float = 1) -> float:
    """
    Calculate the expected volatility of the portfolio.

    Parameters: asset_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. covariance_matrix (numpy.ndarray): Expected covariance matrix.
    This is a 2D array where the element at the i-th row and j-th column represents the covariance between the i-th
    and j-th assets. initial_capital (float, optional): The initial capital invested in the portfolio. Defaults to 1.

    Returns: float: Expected volatility of the portfolio. This is calculated as the square root of the dot product of
    the weights array and the product of the weights array and the covariance matrix.

    This function calculates the portfolio variance by taking the dot product of the asset weights and the product of
    the asset weights and the covariance matrix. The square root of the variance is then returned as the portfolio
    volatility.
    """
    # Calculate the portfolio variance
    pf_variance = portfolio_variance(asset_weights, covariance_matrix, initial_capital)

    # Return the square root of the variance to get the volatility
    return np.sqrt(pf_variance)


import numpy as np
from scipy.optimize import minimize


def sum_to_one_constraint(weights):
    """
    Constraint function for the optimization that ensures the sum of the weights equals 1.
    This is necessary for the portfolio to represent a complete allocation of capital.

    Parameters:
    - weights (np.ndarray): Portfolio weights to be optimized.

    Returns:
    - float: The difference from 1, which should be zero when the constraint is met.
    """
    return np.sum(weights) - 1


def mv_portfolio_objective(weights, expected_returns, covariance_matrix, risk_aversion_factor,
                           transaction_fee_rate, current_weights, initial_capital):
    """
    Objective function for the portfolio optimization that calculates the negative of the adjusted portfolio return.
    This is designed for minimization in an optimizer to maximize the original function.

    Parameters:
    - weights (np.ndarray): Portfolio weights.
    - expected_returns (np.ndarray): Expected returns for each asset.
    - covariance_matrix (np.ndarray): Covariance matrix of the asset returns.
    - risk_aversion_factor (float): Coefficient that scales the importance of the variance penalty.
    - transaction_fee_rate (float): Transaction fee rate applied to the amount traded.
    - current_weights (np.ndarray): Current portfolio weights before rebalancing.
    - initial_capital (float): Total capital, defaults to 1 for percentage allocation.

    Returns:
    - float: The value of the portfolio's objective function, for minimization.
    """
    pf_return = np.dot(weights, expected_returns)  # Calculate portfolio return
    pf_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))  # Calculate portfolio variance
    tr_costs = transaction_fee_rate * np.sum(np.abs(weights - current_weights))  # Transaction costs

    # Objective: Maximize return and minimize variance and transaction costs
    # Multiply variance by 0.5 and risk aversion factor for a balanced objective
    return -(pf_return - initial_capital * risk_aversion_factor * 0.5 * pf_variance - tr_costs)


def mean_variance_portfolio(expected_returns, covariance_matrix, risk_aversion_factor,
                            transaction_fee_rate, bounds, current_weights, initial_capital=1.0):
    """
    Optimizes portfolio using a mean-variance approach including transaction costs.

    Parameters:
    - expected_returns (np.ndarray): Array of expected returns for each asset.
    - covariance_matrix (np.ndarray): Covariance matrix for the returns of the assets.
    - risk_aversion_factor (float): How much risk is the investor willing to take.
    - transaction_fee_rate (float): Rate at which transaction fees are applied.
    - current_weights (np.ndarray): Current distribution of weights across the portfolio.
    - bounds (list of tuples): List of (min, max) pairs for each portfolio weight.
    - initial_capital (float): Initial capital, default is 1, typically implies a normalized portfolio.

    Returns:
    - np.ndarray: The optimized weights for the portfolio.
    """
    num_assets = expected_returns.shape[0]
    # Ensure bounds are repeated for each asset correctly
    bounds = [tuple(bounds)] * num_assets  # Apply the same bounds to all assets if a single bound is given

    # Initial guess (can be the current weights or evenly distributed)
    initial_guess = np.array(current_weights)

    # Define the constraints
    constraints = [{'type': 'eq', 'fun': sum_to_one_constraint}]

    # Optimize the portfolio
    result = minimize(mv_portfolio_objective, initial_guess,
                      args=(expected_returns, covariance_matrix, risk_aversion_factor,
                            transaction_fee_rate, current_weights, initial_capital),
                      method='SLSQP', bounds=bounds, constraints=constraints )

    return result.x



class Position:
    def __init__(self, capital: float, weights: np.ndarray, date: datetime, horizon: datetime,
                 next_weights: Optional[np.ndarray] = None):
        """
        Initializes a new instance of the Position class.

        Args:
            capital (float): Initial capital amount, must be greater than zero.
            weights (np.ndarray): Asset allocation weights, must sum to 1.
            date (datetime): Effective date of this position.
            next_weights (np.ndarray, optional): Future asset allocation weights.
            horizon (int, optional): Investment horizon in days.

        Raises:
            ValueError: If the initial conditions for capital or weights are not met.
        """
        if capital <= 0:
            raise ValueError("Capital must be greater than zero.")
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must sum to 1.")

        self._capital = capital
        self._weights = weights
        self._date = date
        self._next_weights = next_weights
        self._horizon = horizon
        self._returns = None

    @property
    def capital(self) -> float:
        """Returns the capital amount of the position."""
        return self._capital

    @property
    def weights(self) -> np.ndarray:
        """Returns the current asset allocation weights."""
        return self._weights

    @property
    def date(self) -> datetime:
        """Returns the effective date of this position."""
        return self._date

    @property
    def next_weights(self) -> Optional[np.ndarray]:
        """Returns the next planned weights for asset allocation."""
        return self._next_weights

    @property
    def horizon(self) -> datetime:
        """Returns the investment horizon in days."""
        return self._horizon

    @property
    def returns(self) -> np.ndarray:
        """Returns the investment horizon in days."""
        return self._returns

    def update_nweights(self, next_weights: np.ndarray):
        """
        Updates both the next weights and the investment horizon. Validates both before updating.

        Args:
            next_weights (np.ndarray): New next weights, must sum to 1.

        Raises:
            ValueError: If the next weights do not sum to 1 or if the horizon is not a positive integer.
        """
        if not np.isclose(next_weights.sum(), 1):
            raise ValueError("Next weights must sum to 1.")

        self._next_weights = next_weights

    def update_returns(self, returns: np.ndarray):
        if self.weights.shape[0] != returns.shape[0]:
            raise ValueError("The provided returns does not match the expected number of assets.")
        self._returns = returns

    def forward(self, new_horizon: datetime):
        self._capital = 0 # fonction de calcule à definir
        self._weights = self._next_weights
        self._date = self._horizon
        self._next_weights = None
        self._horizon = new_horizon
        self._returns = None
