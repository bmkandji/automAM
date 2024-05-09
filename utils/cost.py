import numpy as np


def trans_cost_imp(x: float, current_weights: np.ndarray,
                   next_weights: np.ndarray, transaction_fee_rate: float) -> float:
    """
    Calculates a modified transaction cost for a given value x, considering dynamic pricing and quantity conditions.

    Parameters:
    - x (float): The independent variable for which the cost is being calculated.
    - current_weights (np.ndarray): (np.ndarray): Current distribution of weights across the portfolio.
    - next_weights (np.ndarray): (np.ndarray): Next distribution of weights across the portfolio.
    - transaction_fee_rate (float): Scaling factor that modifies the cost impact of deviations from optimal conditions.

    Returns:
    - float: The computed transaction cost incorporating conditions and scaled adjustments.
    """
    current_weights, next_weights = current_weights[1:], next_weights[1:]  # Excludes the first element of
    # current_weights and next_weights

    # Calculate conditions for cost adjustments
    condition_ge = next_weights * x >= current_weights  # Condition for x ≥ current_weights_i/next_weights_i
    condition_le = next_weights * x < current_weights  # Condition for x < current_weights_i/next_weights_i

    # Calculate cost components based on conditions
    sum1 = np.sum(transaction_fee_rate * np.abs(current_weights - x * next_weights) * condition_le)  # Cost when x <
    # current_weights_i/next_weights_i
    sum2 = np.sum((transaction_fee_rate / (1 - transaction_fee_rate))
                  * np.abs(current_weights - x * next_weights) * condition_ge)  # Cost when x ≥
    # current_weights_i/next_weights_i

    # Final cost calculation
    return sum1 + sum2 + x - 1


def transact_cost(current_weights: np.ndarray, next_weights: np.ndarray, transaction_fee_rate: float) -> float:
    """
    Identifies an optimal target 'x' that satisfies or minimizes transaction costs based on the function
    transac_cost_imp.

    Parameters:
    - current_weights (np.ndarray): Vector of prices, including an initial element that is adjusted in calculations.
    - next_weights (np.ndarray): Vector of quantities, where zeros are replaced by an arbitrary value greater than one.
    - transaction_fee_rate (float): Scaling factor used in the cost function to balance the impact of deviations.

    Returns:
    - float: The optimal 'x' value that potentially minimizes the transaction costs.
    """
    if np.all(next_weights[1:] == 0):  # Check if all next_weights values excluding the 
        # (à voir or len(next_weights[1:]) == 0 )
        # first are zero
        return 1 - np.sum(transaction_fee_rate * current_weights[1:])  # If true, returns the sum of weightsed prices

    alter = next_weights.copy()  # Copies next_weights to avoid modifying the original array
    alter[alter == 0] = 2  # Replaces zeros in next_weights with an arbitrary value > 1 to avoid division by zero

    breaks_pt = (current_weights / alter)[1:]  # Calculates potential break points from price and adjusted quantity
    # arrays
    target_pt = np.sort(breaks_pt[breaks_pt < 1])[::-1]  # Selects and sorts feasible targets

    if len(target_pt) == 0:
        return 1 - (1 - np.sum(transaction_fee_rate * current_weights[1:]))/(1 - np.sum(
            transaction_fee_rate * next_weights[1:]))

    target_pt = np.insert(target_pt, 0, 1)  # Inserts 1 at the beginning of targets array

    img_tg_pt = [trans_cost_imp(target_pt[0], current_weights, next_weights, transaction_fee_rate)]  # Calculates
    # initial cost for the first target

    i, max_iter = 1, len(target_pt)  # Initializes counter and max iterations

    # Iteratively calculate costs to find the target x minimizing the transaction cost
    while img_tg_pt[-1] > 0 and i < max_iter:
        img_tg_pt.append(trans_cost_imp(target_pt[i], current_weights, next_weights, transaction_fee_rate))
        i += 1
    if img_tg_pt[-1] == 0:
        return 1 - target_pt[i - 1]
    # Calculates target x using the intercept theorem when there's a change in sign of the transaction cost
    fee = 1 - (target_pt[i - 1] * img_tg_pt[-2] - target_pt[i - 2] * img_tg_pt[-1]) / (img_tg_pt[-2] - img_tg_pt[-1])

    return fee
