---
---

# FinancialStatsPython

**FinancialStatsPython** is a Python project designed to perform detailed financial analyses of stocks and market indices. This script automates data collection, risk analysis, interactive chart generation, and price forecasting by combining advanced statistical techniques with artificial intelligence. The goal is to provide a comprehensive tool for market performance evaluation and risk management.

---
---

## Key Features:

### **Data Collection**
- Automatic retrieval of historical data from Yahoo Finance.
- Calculation of key metrics such as average price, volumes, and daily returns.

### **Risk and Volatility Analysis**
- Calculation of indicators such as Volatility, Value at Risk (VaR), Expected Shortfall (CVaR), Sharpe Ratio, and more.
- Analysis of maximum drawdown to evaluate the largest losses during the analyzed period.

### **Forecasting Models**
- Utilization of ARIMA and GARCH models for price and volatility forecasting.
- Implementation of an AI model (NOT STABLE) for advanced future price predictions.

### **Benchmark Analysis**
- Comparison of stocks against reference benchmarks, including beta and correlation calculations.
- Evaluation of cointegration with benchmarks for long-term analysis.

### **Interactive Visualizations**
- Generation of interactive charts with Plotly and direct terminal representations.
- Seasonal decomposition to identify trends, seasonality, and residuals.

### **HTML Dashboard**
- Automated creation of interactive dashboards to explore results and charts.

---

## Technologies Used:

## - **Python Libraries:** 
- `pandas==2.2.3`
- `plotly==5.24.1`
- `plotext==5.3.2`
- `yfinance==0.2.51`
- `numpy==2.0.2`
- `scipy==1.15.0`
- `jinja2==3.1.5`
- `statsmodels==0.14.4`
- `tensorflow==2.18.0`
- `absl-py==2.1.0`
- `deep-translator==1.11.4`
- `matplotlib==3.10.0`
- `requests==2.32.3`
- `arch==7.2.0`
- `scikit-learn==1.6.1`
- `keras==3.8.0`

## - **Data Source:** 
- `Yahoo Finance API`

## - **Models:** 
- `ARIMA, GARCH, LSTM`
---
---

# Example Output
---
## Dashboard Html

[https://github.com/fr-cm/FinancialStatsPython/blob/Example/EN_code/AssetsEX/AAPL_dashboard.html](https://fr-cm.github.io/FinancialStatsPythonResults/Website/IONQ_dashboard.html)


## Terminale

[![Output](https://github.com/fr-cm/FinancialStatsPython/blob/d73688ae1ee843dd5ced2e75691b766f7bc138f0/EN_code/AssetsEX/Example_output_on_terminal_EN.png)]

---



## Disclaimer:

This project is for educational purposes only, it does not constitute financial advice. The author explicitly disclaims any liability for errors, omissions, or inaccuracies in the analyses and information provided. The use of this data is at the user's own risk. The author does not guarantee the accuracy, completeness, or reliability of the information presented or the absence of malfunctions in the script used. Furthermore, the author will not be held responsible for any direct, indirect, incidental, or consequential damages arising from the use or reliance on these analyses. Users are advised to independently verify and validate the information before making any decisions based on it.

---
---
