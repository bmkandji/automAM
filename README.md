# Automated Asset Management with R and Python

## Overview

This project aims to automate asset management tasks using a combination of R and Python. It leverages R for advanced statistical modeling and time series analysis, while Python handles data manipulation, preprocessing, and interfacing with external APIs.

## Prerequisites

- Python 3.12.0
- R (recommended version: 4.3.1)

## Features

- **Data Retrieval:** Fetches historical financial data for specified assets from online sources such as Yahoo Finance, Alpha Vantage, or Alpaca Markets.
- **Data Processing:** Cleans and preprocesses the retrieved data, handling missing values.
- **Statistical Modeling:** Utilizes R's powerful statistical packages like `rmgarch` for fitting GARCH models or machine learning models like LSTMs to conduct volatility analysis.
- **Portfolio Optimization:** Implements portfolio optimization techniques to construct efficient portfolios based on risk and expected return.
- **Automated Reporting:** Generates reports summarizing portfolio performance, risk metrics, and investment recommendations.

## Setup

1. **Environment Configuration:**
   - Set up a virtual environment for the project using Python 3.12.0.
   - Install the required Python packages listed in `requirements.txt`.
   - Set the `R_HOME` environment variable to the R installation directory.

2. **Installation:**

   ```bash
   pip install -r requirements.txt
   ```

3. **R Integration:**
   - Ensure R is installed on your system.
   - Install required R packages (`rmgarch`, `rugarch`, etc.) using the R console or package manager.

## License

This project is licensed under the [MIT License](LICENSE).
