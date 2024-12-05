# Automated Asset Management with R and Python

## Overview
This project aims to automate asset management tasks using a combination of R and Python. It leverages R for advanced statistical modeling and time series analysis, while Python handles data manipulation, preprocessing, and interfacing with external APIs.

## Prerequisites

- Python 3.12.0
- R (recommended version: 4.3.1)
- 
## Features
- **Data Retrieval:** Fetches historical financial data for specified assets from online sources such as Yahoo Finance or Alpha Vantage.
- **Data Processing:** Cleans and preprocesses the retrieved data, handling missing values, and adjusting for stock splits and dividends.
- **Statistical Modeling:** Utilizes R's powerful statistical packages like `rmgarch` for fitting GARCH models and conducting volatility analysis.
- **Portfolio Optimization:** Implements portfolio optimization techniques to construct efficient portfolios based on risk and return objectives.
- **Automated Reporting:** Generates reports summarizing portfolio performance, risk metrics, and investment recommendations.

## Setup
1. **Environment Configuration:**
    - Set up a virtual environment for the project using Python 3.12.10.
    - Install the required Python packages listed in `requirements.txt`.
    - Set the `R_HOME` environment variable to the R installation directory.
    
2. **Installation:**
    ```bash
    pip install -r requirements.txt
    ```


3. **R Integration:**
    - Ensure R is installed on your system.
    - Install required R packages (`rmgarch`, `rugarch`, etc.) using the R console or package manager.

## Usage
1. **Data Retrieval:**
    - Use `data_fetcher.py` to fetch historical financial data for specified assets.
    
2. **Data Processing:**
    - Preprocess the retrieved data using `data_processor.py`.
    
3. **Statistical Modeling:**
    - Fit GARCH models and conduct volatility analysis with `model.py`.
    
4. **Portfolio Optimization:**
    - Utilize portfolio optimization techniques in `portfolio.py` to construct efficient portfolios.

## Contributing
Contributions to this project are welcome! If you have any ideas, feature requests, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
