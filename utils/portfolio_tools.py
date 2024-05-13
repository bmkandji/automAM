import numpy as np
from scipy.optimize import minimize, LinearConstraint
from utils.cost import transact_cost


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


def portfolio_return(next_weights, current_weights: np.ndarray, transaction_fee_rate,
                     expected_returns: np.ndarray, initial_capital: float = 1,
                     scale: float = 100, tk_acount_capital: bool = False,
                     tk_acount_scale: bool = False) -> float:
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
    check_inputs(current_weights, expected_returns, initial_capital)
    if not tk_acount_capital:
        initial_capital = 1
    if not tk_acount_scale:
        scale = 1
    fee = transact_cost(current_weights, next_weights, transaction_fee_rate)
    net_return = (initial_capital / scale) * (
            (1 - fee) * np.dot(next_weights, expected_returns) - fee)  # Calculate the net portfolio return
    return net_return


def portfolio_variance(next_weights, current_weights: np.ndarray,
                       transaction_fee_rate, covariance_matrix: np.ndarray,
                       initial_capital: float = 1, scale: float = 100, tk_acount_capital: bool = False,
                       tk_acount_scale: bool = False) -> float:
    """
    Calculate the expected variance of the portfolio.
    Parameters: current_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. covariance_matrix (numpy.ndarray): Expected covariance matrix.
    This is a 2D array where the element at the i-th row and j-th column represents the covariance between the i-th
    and j-th assets. initial_capital (float, optional): The initial capital invested in the portfolio. Defaults to 1.

    Returns: float: Expected variance of the portfolio. This is calculated as the dot product of the weights array
    and the product of the weights array and the covariance matrix.

    This function calculates the portfolio variance by taking the dot product of the asset weights and the product of
    the asset weights and the covariance matrix. The variance is scaled by the square of the initial capital.
    """
    check_inputs(current_weights, covariance_matrix, initial_capital)

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be a square matrix.")
    if not tk_acount_capital:
        initial_capital = 1
    if not tk_acount_scale:
        scale = 1

    fee = transact_cost(current_weights, next_weights, transaction_fee_rate)
    pf_variance = (initial_capital / scale) ** 2 * (1 - fee) ** 2 * np.dot(next_weights.T,
                                                                           np.dot(covariance_matrix, next_weights))

    return pf_variance


def portfolio_variance_mix(next_weights, current_weights: np.ndarray,
                           next_weights_ref, current_weights_ref: np.ndarray,
                           transaction_fee_rate, covariance_matrix: np.ndarray,
                           initial_capital: float = 1, scale: float = 100,
                           tk_acount_capital: bool = False,
                           tk_acount_scale: bool = False) -> float:
    """
    Calculate the expected variance of the portfolio.
    Parameters: current_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
    portfolio invested in the corresponding asset. covariance_matrix (numpy.ndarray): Expected covariance matrix.
    This is a 2D array where the element at the i-th row and j-th column represents the covariance between the i-th
    and j-th assets. initial_capital (float, optional): The initial capital invested in the portfolio. Defaults to 1.

    Returns: float: Expected variance of the portfolio. This is calculated as the dot product of the weights array
    and the product of the weights array and the covariance matrix.

    This function calculates the portfolio variance by taking the dot product of the asset weights and the product of
    the asset weights and the covariance matrix. The variance is scaled by the square of the initial capital.
    """
    check_inputs(current_weights, covariance_matrix, initial_capital)

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be a square matrix.")
    if not tk_acount_capital:
        initial_capital = 1
    if not tk_acount_scale:
        scale = 1

    fee = transact_cost(current_weights, next_weights, transaction_fee_rate)
    fee_ref = transact_cost(current_weights_ref, next_weights_ref, transaction_fee_rate)
    variance_mix = ((initial_capital / scale) ** 2
                    * np.dot(((1 - fee) * next_weights - (1 - fee_ref) * next_weights_ref).T,
                             np.dot(covariance_matrix, ((1 - fee) * next_weights - (1 - fee_ref)
                                                        * next_weights_ref))))

    return variance_mix


def portfolio_volatility(next_weights, current_weights: np.ndarray,
                         transaction_fee_rate, covariance_matrix: np.ndarray,
                         initial_capital: float = 1, scale: float = 100, tk_acount_capital: bool = False,
                         tk_acount_scale: bool = False) -> float:
    """
    Calculate the expected volatility of the portfolio.

    Parameters: current_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
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
    pf_variance = portfolio_variance(next_weights, current_weights, transaction_fee_rate, covariance_matrix,
                                     initial_capital, scale, tk_acount_capital, tk_acount_scale)

    # Return the square root of the variance to get the volatility
    return np.sqrt(pf_variance)


def portfolio_volatility_mix(next_weights, current_weights: np.ndarray,
                             next_weights_ref, current_weights_ref,
                             transaction_fee_rate, covariance_matrix: np.ndarray,
                             initial_capital: float = 1, scale: float = 100, tk_acount_capital: bool = False,
                             tk_acount_scale: bool = False) -> float:
    """
    Calculate the expected volatility of the portfolio.

    Parameters: current_weights (numpy.ndarray): Array of asset weights. Each element represents the proportion of the
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
    pf_variance = portfolio_variance_mix(next_weights, current_weights,
                                         next_weights_ref, current_weights_ref,
                                         transaction_fee_rate, covariance_matrix,
                                         initial_capital, scale, tk_acount_capital, tk_acount_scale)

    # Return the square root of the variance to get the volatility
    return np.sqrt(pf_variance)


def fw_portfolio_value(asset_weights: np.ndarray, returns: np.ndarray,
                       initial_capital: float = 1, scale: float = 100) -> float:
    """
    Calculate the final value of a portfolio based on the asset weights, expected returns,
    and the initial capital invested.

    Parameters:
    - asset_weights (np.ndarray): An array of weights for each asset in the portfolio,
                                  where each weight are a fraction of the total investment.
    - returns (np.ndarray): An array of observed logarithmic returns for each asset over the period considered.
    - initial_capital (float, optional): The total initial capital invested in the portfolio, default is 1.

    Returns:
    - float: The final total value of the portfolio after applying the expected returns.

    This function uses the exponential of the expected returns to determine the growth factors for each asset,
    then multiplies these by the respective portions of the initial capital according to the asset weights,
    and sums the results to obtain the final portfolio value.
    """

    # Validate input lengths
    if len(asset_weights) != len(returns):
        raise ValueError("Asset weights and expected returns must have the same length.")

    # Calculate growth factors from expected logarithmic returns
    growth_factors = np.exp(returns / scale)
    portfolio = initial_capital * asset_weights * growth_factors
    pf_value = np.sum(portfolio)
    weights = portfolio / pf_value

    return pf_value, weights


def capital_fw(weights: np.ndarray, current_weights: np.ndarray, transaction_fee_rate: float,
               initial_capital: float = 1.0) -> np.ndarray:
    """
    Calculate the final capital after rebalancing the portfolio considering the transaction costs.

    Parameters:
    - weights (cvxpy.Variable): The new weights of the assets in the portfolio being optimized.
    - current_weights (np.ndarray): Current portfolio weights before rebalancing.
    - transaction_fee_rate (float): The transaction fee rate as a percentage of the traded amount.
    - initial_capital (float): Total capital of the portfolio at the start.

    Returns:
        object:
    - cvxpy.Expression: The total capital after deducting transaction costs.
    """
    return initial_capital - transact_cost(current_weights, weights, transaction_fee_rate)


def merge_weights(weights: np.ndarray, fixed_weights: np.ndarray) -> np.ndarray:
    """
    Combines dynamic weights with fixed weights based on a mask defined in fixed_weights.

    Parameters:
    - weights (np.ndarray): The dynamic weights for the assets where weights are not fixed.
    - fixed_weights (np.ndarray): A two-row array where the first row is a mask (1 for fixed, 0 for not fixed)
                                  and the second row contains the actual fixed weights.

    Returns:
    - np.ndarray: The merged weights for all assets including both fixed and dynamic weights.
    """
    num_assets = fixed_weights[0].shape[0]
    all_weights = np.zeros(num_assets)

    # Error checking
    if weights.size != sum(fixed_weights[0] == 0):
        raise ValueError("Number of dynamic weights does not match the number of non-fixed positions.")

    # Assign fixed weights based on the mask
    fixed_mask = fixed_weights[0] == 1
    all_weights[fixed_mask] = fixed_weights[1][fixed_mask]

    # Assign dynamic weights to the non-fixed positions
    non_fixed_mask = fixed_weights[0] == 0
    all_weights[non_fixed_mask] = weights

    return all_weights


def sum_to_one_constraint(weights: np.ndarray, fixed_weights: np.ndarray) -> float:
    """
    Constraint function for optimization that ensures the sum of the combined (fixed and dynamic) weights equals 1.

    Parameters:
    - weights (np.ndarray): The dynamic portion of the portfolio weights to be optimized.
    - fixed_weights (np.ndarray): Array specifying which weights are fixed and their values.

    Returns:
    - float: The deviation from 1, which should be zero if the constraint is met.
    """
    return np.sum(merge_weights(weights, fixed_weights)) - 1


def mv_portfolio_objective(weights: np.ndarray, expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray, risk_aversion_factor: float,
                           transaction_fee_rate: float, current_weights: np.ndarray,
                           initial_capital: float = 1.0, scale: float = 100,
                           tk_acount_capital: bool = False, tk_acount_scale: bool = False) -> float:
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
    if not tk_acount_capital:
        initial_capital = 1
    if not tk_acount_scale:
        scale = 1
    fee = transact_cost(current_weights, weights, transaction_fee_rate)
    net_return = (1 - fee) * np.dot(weights, expected_returns) - fee  # Calculate the net portfolio return
    pf_variance = (1 - fee) ** 2 * np.dot(weights.T, np.dot(covariance_matrix, weights))  # Calculate portfolio variance

    # Objective: Maximize return and minimize variance and transaction costs
    # Multiply variance by 0.5 and risk aversion factor for a balanced objective
    return -(risk_aversion_factor * net_return - scale * initial_capital * 0.5 * pf_variance)


def mean_variance_portfolio(expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                            risk_aversion_factor: float, transaction_fee_rate: float, bounds: list,
                            current_weights: np.ndarray, initial_capital: float = 1.0,
                            scale: float = 100, fixed_weights: np.ndarray = None,
                            tk_acount_capital: bool = False, tk_acount_scale: bool = False) -> np.ndarray:
    """
    Optimizes a portfolio using a mean-variance approach, including transaction costs,
    with the ability to specify fixed weights for certain assets. The optimization applies
    the same bounds to all tradable assets.

    Parameters:
    - expected_returns (np.ndarray): Expected returns for each asset.
    - covariance_matrix (np.ndarray): Covariance matrix for asset returns.
    - risk_aversion_factor (float): Degree of risk aversion.
    - transaction_fee_rate (float): Transaction fee rate.
    - current_weights (np.ndarray): Current weights in the portfolio.
    - bounds (list of tuple): A single tuple containing (min, max) bounds to be applied to all tradable assets.
    - initial_capital (float, optional): Initial capital, defaults to 1. Typically, represents a normalized portfolio.
    - scale (float, optional): Scaling factor for the optimization, defaults to 100.
    - fixed_weights (np.ndarray, optional): A 2-row array where the first row indicates whether a weights are fixed
    (1) or not (0), and the second row specifies the fixed weights.

    Returns:
    - np.ndarray: Optimized portfolio weights.
    """
    num_assets = expected_returns.shape[0]

    if fixed_weights is None:
        fixed_weights = np.zeros((2, num_assets))

    # Number of assets not fixed
    num_traded_assets = np.sum(fixed_weights[0] == 0)

    # Apply the same bounds to all tradable assets
    bounds = [tuple(bounds)] * num_traded_assets  # Ensure bounds are repeated for each tradable asset correctly

    # Initial guess for tradable assets, filtering out the fixed weights
    initial_guess = current_weights[fixed_weights[0] == 0]

    # Define the constraint that ensures the sum of weights equals 1, taking into account fixed weights
    constraints = [{'type': 'eq', 'fun': lambda x: sum_to_one_constraint(x, fixed_weights)}]

    # Define the objective function, adjusting it for fixed weights
    def objective(x):
        return mv_portfolio_objective(merge_weights(x, fixed_weights),
                                      expected_returns, covariance_matrix,
                                      risk_aversion_factor, transaction_fee_rate,
                                      current_weights, initial_capital,
                                      scale, tk_acount_capital, tk_acount_scale)

    # Optimize the portfolio
    result = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    # Combine the optimized tradable weights with the fixed weights and return
    return merge_weights(result.x, fixed_weights)


def max_return(expected_returns: np.ndarray, covariance_matrix: np.ndarray,
               max_vol: float, transaction_fee_rate: float,
               bounds: list, current_weights: np.ndarray,
               initial_capital: float = 1.0, scale: float = 100, fixed_weights: np.ndarray = None,
               tk_acount_capital: bool = False, tk_acount_scale: bool = False) -> np.ndarray:
    """
    Optimizes a portfolio using a mean-variance approach, including transaction costs,
    with the ability to specify fixed weights for certain assets. The optimization applies
    the same bounds to all tradable assets.

    Parameters:
    - expected_returns (np.ndarray): Expected returns for each asset.
    - covariance_matrix (np.ndarray): Covariance matrix for asset returns.
    - risk_aversion_factor (float): Degree of risk aversion.
    - transaction_fee_rate (float): Transaction fee rate.
    - current_weights (np.ndarray): Current weights in the portfolio.
    - bounds (list of tuple): A single tuple containing (min, max) bounds to be applied to all tradable assets.
    - initial_capital (float, optional): Initial capital, defaults to 1. Typically, represents a normalized portfolio.
    - scale (float, optional): Scaling factor for the optimization, defaults to 100.
    - fixed_weights (np.ndarray, optional): A 2-row array where the first row indicates whether a weights are fixed
    (1) or not (0), and the second row specifies the fixed weights.

    Returns:
    - np.ndarray: Optimized portfolio weights.
    """
    num_assets = expected_returns.shape[0]

    if fixed_weights is None:
        fixed_weights = np.zeros((2, num_assets))

    # Number of assets not fixed
    num_traded_assets = np.sum(fixed_weights[0] == 0)

    # Apply the same bounds to all tradable assets
    bounds = [tuple(bounds)] * num_traded_assets  # Ensure bounds are repeated for each tradable asset correctly

    # Initial guess for tradable assets, filtering out the fixed weights
    initial_guess = current_weights[fixed_weights[0] == 0]

    # Define the constraint that ensures the sum of weights equals 1, taking into account fixed weights
    constraints = [{'type': 'eq', 'fun': lambda x: sum_to_one_constraint(x, fixed_weights)},
                   {'type': 'ineq', 'fun': lambda x:
                   max_vol - portfolio_volatility(merge_weights(x, fixed_weights),
                                                  current_weights, transaction_fee_rate,
                                                  covariance_matrix, initial_capital, scale,
                                                  tk_acount_capital, tk_acount_scale)
                    }
                   ]

    # Define the objective function, adjusting it for fixed weights
    def objective(x):
        return - portfolio_return(merge_weights(x, fixed_weights), current_weights,
                                  transaction_fee_rate, expected_returns,
                                  initial_capital, scale, tk_acount_capital, tk_acount_scale)

    # Optimize the portfolio
    result = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    # Combine the optimized tradable weights with the fixed weights and return
    return merge_weights(result.x, fixed_weights)


def tracking_error(expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                   ref_portfolio: dict, tol: float, transaction_fee_rate: float,
                   bounds: list, current_weights: np.ndarray,
                   initial_capital: float = 1.0, scale: float = 100, fixed_weights: np.ndarray = None,
                   tk_acount_capital: bool = False, tk_acount_scale: bool = False) -> np.ndarray:
    """
    Optimizes a portfolio using a mean-variance approach, including transaction costs,
    with the ability to specify fixed weights for certain assets. The optimization applies
    the same bounds to all tradable assets.

    Parameters:
    - expected_returns (np.ndarray): Expected returns for each asset.
    - covariance_matrix (np.ndarray): Covariance matrix for asset returns.
    - risk_aversion_factor (float): Degree of risk aversion.
    - transaction_fee_rate (float): Transaction fee rate.
    - current_weights (np.ndarray): Current weights in the portfolio.
    - bounds (list of tuple): A single tuple containing (min, max) bounds to be applied to all tradable assets.
    - initial_capital (float, optional): Initial capital, defaults to 1. Typically, represents a normalized portfolio.
    - scale (float, optional): Scaling factor for the optimization, defaults to 100.
    - fixed_weights (np.ndarray, optional): A 2-row array where the first row indicates whether a weights are fixed
    (1) or not (0), and the second row specifies the fixed weights.

    Returns:
    - np.ndarray: Optimized portfolio weights.
    """
    num_assets = expected_returns.shape[0]

    if fixed_weights is None:
        fixed_weights = np.zeros((2, num_assets))

    # Number of assets not fixed
    num_traded_assets = np.sum(fixed_weights[0] == 0)

    # Apply the same bounds to all tradable assets
    bounds = [tuple(bounds)] * num_traded_assets  # Ensure bounds are repeated for each tradable asset correctly

    # Initial guess for tradable assets, filtering out the fixed weights
    initial_guess = current_weights[fixed_weights[0] == 0]

    A_eq = np.array([[1] * num_traded_assets])
    b_eq = np.array([1 - sum(fixed_weights[1][fixed_weights[0] == 1])])
    linear_constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)

    # Define the constraint that ensures the sum of weights equals 1, taking into account fixed weights
    constraints = [linear_constraint_eq,
                   {'type': 'ineq', 'fun': lambda x:
                   tol - portfolio_volatility_mix(merge_weights(x, fixed_weights),
                                                  current_weights, ref_portfolio["next_weights"],
                                                  ref_portfolio["weights"], transaction_fee_rate,
                                                  covariance_matrix, initial_capital, scale,
                                                  tk_acount_capital, tk_acount_scale)}
                   ]

    # Define the objective function, adjusting it for fixed weights
    def objective(x):
        return (portfolio_return(ref_portfolio["next_weights"], ref_portfolio["weights"],
                                 transaction_fee_rate, expected_returns,
                                 initial_capital, scale, tk_acount_capital, tk_acount_scale)
                - portfolio_return(merge_weights(x, fixed_weights), current_weights,
                                   transaction_fee_rate, expected_returns,
                                   initial_capital, scale, tk_acount_capital, tk_acount_scale))

    # Optimize the portfolio
    result = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    # Combine the optimized tradable weights with the fixed weights and return
    return merge_weights(result.x, fixed_weights)
