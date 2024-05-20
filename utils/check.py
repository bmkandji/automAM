import numpy as np


def checks_weights(weights: np.ndarray):
    if not np.isclose(weights.sum(), 1):
        raise ValueError("weights must sum to 1.")


def check_configs(portfolio=None, data=None,
                  model=None, rportfolio=None,
                  check_date: bool = True,
                  check_scale: bool = False,
                  check_fit_date: bool = True):
    """
    Validates the consistency of configuration details among any provided combinations
    of portfolio, data, and model configurations. This function checks if all provided
    configurations use the same set of symbols and, where applicable, checks for date
    alignment between the configurations. At least two configurations
    must be provided to conduct a comparison.

    Parameters:
    portfolio (Portfolio, optional): An instance containing portfolio-specific configurations.
    data (Data, optional): An instance containing data-specific configurations.
    model (Model, optional): An instance containing model-specific configurations.

    Raises:
    ValueError: If fewer than two configurations are provided or if there is a mismatch in symbols
                or dates among the provided configurations.
    """
    # Check if at least two configurations are provided
    provided_configs = sum([1 for cfg in [portfolio, data, model, rportfolio] if cfg is not None])
    if provided_configs < 2:
        raise ValueError("Configuration mismatch: Insufficient configurations provided for comparison.")

    # Prepare symbol sets from each provided configuration
    portfolio_symbols = set(portfolio.pf_config["symbols"]) if portfolio else None
    data_symbols = set(data.data_config["symbols"]) if data else None
    model_symbols = set(model.model_config["symbols"]) if model else None
    rportfolio_symbols = set(rportfolio.pf_config["symbols"]) if rportfolio else None
    # Check if all provided symbol sets are identical
    symbol_sets = [s for s in [portfolio_symbols, data_symbols, model_symbols, rportfolio_symbols] if s is not None]
    if not all(s == symbol_sets[0] for s in symbol_sets):
        raise ValueError("Configuration mismatch: symbol sets do not align.")

    # Check date alignment among all configurations if check_date is True
    if check_date:
        # Prepare date values from each provided configuration
        portfolio_date = portfolio.date if portfolio else None
        data_date = data.data_config.get("end_date") if data else None
        model_date = model.metrics.get("fit_date") if model else None
        # Collect non-None dates
        dates = [d for d in [portfolio_date, data_date, model_date] if d is not None]

        # Check if all provided dates are the same
        if len(set(dates)) > 1 or len(dates) < 2:
            raise ValueError("Configuration mismatch: dates do not align among provided configurations.")

    if check_scale and model and data and (model.metrics['scale'] != data.data_config["scale"]):
        raise ValueError("model and data do not have the same scale value")

    if check_fit_date and portfolio and data and (portfolio.date != data.metrics["fit_date"]):
        raise ValueError("portfolio and data do not have the same scale value")
