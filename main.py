########### TEST PM ##############
from utils.load import load_json_config
from src.pfManager import PortfolioManager  # Replace with the correct import


def main(pm_config: dict):
    # Create an instance of the PortfolioManager
    bot = PortfolioManager(pm_config)

    # Start the portfolio management process
    bot.start()


def main_bis():
    # Create an instance of the PortfolioManager
    pm_config = load_json_config(r"src/pfManger_settings/pfPaperMananger_settings.json")

    bot = PortfolioManager(pm_config)

    # Start the portfolio management process
    bot.start()

"""
from utils.input import interface_input

if __name__ == '__main__':
    # Load the configuration for the Portfolio Manager
    input_data = {"selected_assets": ["Default Settings"], "selected_strategy": "bis", "aversion": "", "tol": 10000, "ref_portfolio": "Eq_weighted"}
    pm_config = load_json_config(r"src/pfManger_settings/pfPaperMananger_settings.json")
    pm_config = interface_input(pm_config, input_data)

    main(pm_config)
"""
