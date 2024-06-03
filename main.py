########### TEST PM ##############
from utils.load import load_json_config
from src.pfManager import PortfolioManager  # Replace with the correct import


def main():
    # Load the configuration for the Portfolio Manager
    pm_config = load_json_config(r"src/pfManger_settings/pfPaperMananger_settings.json")

    # Create an instance of the PortfolioManager
    bot = PortfolioManager(pm_config)

    # Start the portfolio management process
    bot.start()


if __name__ == '__main__':
    main()
