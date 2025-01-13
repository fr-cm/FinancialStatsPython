import pandas as pd
from scipy.stats import linregress, norm
from statsmodels.tsa.stattools import coint, adfuller

########################################################################################################
########################################################################################################
#                                            beta                                                      #
########################################################################################################
########################################################################################################
def calculate_beta(data, benchmark_data):
    """
    Calculates the beta of a stock relative to a benchmark.

    Parameters:
        data (pd.DataFrame): DataFrame containing the stock's daily returns in a column named 'daily_returns'.
        benchmark_data (pd.DataFrame): DataFrame containing the benchmark's daily returns in a column named 'daily_returns'.

    Returns:
        float: The beta value of the stock relative to the benchmark, or None if an error occurs.
    """
    if benchmark_data is None or benchmark_data.empty:
        print("    : Error: Invalid or missing benchmark data.")
        return None

    # Merge stock and benchmark data on daily returns
    merged = pd.DataFrame({
        'Stock': data['daily_returns'],
        'Benchmark': benchmark_data['daily_returns']
    }).dropna()



########################################################################################################
########################################################################################################
#                                            correlation cointegration                                 #
########################################################################################################
########################################################################################################
def calculate_correlation_and_cointegration(stock_data, benchmark_data):
    # Align the daily_returns
    returns_stock = stock_data['daily_returns'].dropna()
    returns_bench = benchmark_data['daily_returns'].dropna()
    returns_stock, returns_bench = returns_stock.align(returns_bench, join='inner')

    # Align the 'close' prices
    close_stock = stock_data['close'].dropna()
    close_bench = benchmark_data['close'].dropna()
    close_stock, close_bench = close_stock.align(close_bench, join='inner')

    # Calculate the correlation on returns
    correlation = returns_stock.corr(returns_bench)

    # Calculate the cointegration on prices
    score, pvalue, _ = coint(close_stock, close_bench)

    return correlation, pvalue
