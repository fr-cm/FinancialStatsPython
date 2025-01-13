import numpy as np
from scipy.stats import linregress, norm
import logging
import pandas as pd


########################################################################################################
########################################################################################################
#                                            Volatility                                                #
########################################################################################################
########################################################################################################
def calculate_volatility(data):
    """
    Calculates the historical volatility (standard deviation of returns) annualized.
    """
    volatility = data['daily_returns'].std() * np.sqrt(252) * 100  # Annualized and in percentage
    return volatility



########################################################################################################
########################################################################################################
#                                            RSI                                                       #
########################################################################################################
########################################################################################################

def calculate_rsi(data, period=14):
    """
    Calculates the Relative Strength Index (RSI) without overwriting existing columns.
    """
    if 'rsi' in data.columns:
        print("Warning: The 'rsi' column already exists. Skipping addition.")
        return data
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

# Configure logging
logging.basicConfig(
    filename='../../analisi.log',  # Log file
    filemode='a',  # Append mode
    level=logging.DEBUG,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)




########################################################################################################
########################################################################################################
#                                            Calculate Bollinger Bands                                 #
########################################################################################################
########################################################################################################
def calculate_bollinger_bands(data, period=20):
    """
    Calculates the Bollinger Bands without overwriting existing columns.
    """
    if 'bollinger_upper' in data.columns or 'bollinger_lower' in data.columns:
        print("Warning: The 'bollinger_upper' or 'bollinger_lower' columns already exist. Skipping addition.")
        return data
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    data['bollinger_upper'] = sma + (std * 2)
    data['bollinger_lower'] = sma - (std * 2)
    return data



########################################################################################################
########################################################################################################
#                                            Max Drawdown                                              #
########################################################################################################
########################################################################################################
def calculate_max_drawdown(data):
    """
    Calculates the maximum drawdown.
    """
    cumulative_max = data['close'].cummax()
    drawdown = (data['close'] - cumulative_max) / cumulative_max
    return drawdown.min().item()



########################################################################################################
########################################################################################################
#                                            Sharpe Ratio                                              #
########################################################################################################
########################################################################################################
def calculate_sharpe_ratio(data, risk_free_rate=0.01):
    """
    Calculates the Sharpe ratio.
    risk_free_rate: Annualized risk-free rate.
    """
    excess_returns = data['daily_returns'] - (risk_free_rate / 252)
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio



########################################################################################################
########################################################################################################
#                                            Value at Risk                                             #
########################################################################################################
########################################################################################################
def calculate_var(data, confidence_level=0.95):
    """
    Calculates the Value at Risk (VaR) at the 95% confidence level.
    """
    mean = np.mean(data['daily_returns'].dropna())
    std_dev = np.std(data['daily_returns'].dropna())
    var = norm.ppf(1 - confidence_level) * std_dev - mean
    return var


###############################################################################
#                   1) Expected Shortfall / CVaR (Historical)                 #
###############################################################################
def calculate_expected_shortfall(data, confidence_level=0.95):
    """
    Calculates the Expected Shortfall (or CVaR) using a historical approach.
    confidence_level: confidence level, e.g., 0.95 for 95%.
    float, estimate of the Expected Shortfall (CVaR) as a value (return).
             If you want a positive number expressing 'loss', you can take the absolute value.
    """
    returns = data['daily_returns'].dropna().values
    # Sort the returns from lowest to highest
    sorted_returns = np.sort(returns)

    # Index corresponding to the cutoff (5% if confidence_level=95%)
    cutoff_index = int((1 - confidence_level) * len(sorted_returns))
    if cutoff_index < 1:
        cutoff_index = 1  # in case there is not enough data

    # Historical VaR (at the 5th percentile if confidence_level=95%)
    var_historical = sorted_returns[cutoff_index]

    # Select all returns worse than or equal to VaR
    tail_losses = sorted_returns[:cutoff_index]

    # If there are not enough tail data, avoid division by zero
    if len(tail_losses) == 0:
        return var_historical

    # Expected Shortfall = average of the worst tail returns
    cvar = tail_losses.mean()

    return cvar



###############################################################################
#                            2) Sortino Ratio                                  #
###############################################################################
def calculate_sortino_ratio(data, risk_free_rate=0.01, target_return=0.0):
    """
    Calculates the Sortino Ratio.
    risk_free_rate: Annual risk-free rate (e.g., 0.01 -> 1%).
    target_return: Target return (often 0, or risk_free).
    """
    # Converts the annual risk-free rate to daily (assuming 252 trading days)
    daily_rf = risk_free_rate / 252.0

    # Excess return over the target (or risk-free)
    excess_returns = data['daily_returns'] - daily_rf - target_return

    # Downside deviation (standard deviation only when excess_return < 0)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        # WARNING!! If there are no negative values, Sortino tends to infinity
        return np.inf

    downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

    # Mean annual excess return
    mean_excess_annual = (excess_returns.mean() * 252)
    # Annualized downside deviation
    downside_deviation_annual = downside_deviation * np.sqrt(252)

    sortino_ratio = mean_excess_annual / downside_deviation_annual
    return sortino_ratio


###############################################################################
#                             3) Calmar Ratio                                  #
###############################################################################
def calculate_calmar_ratio(data):
    """
    Calculates the Calmar Ratio = (Average Annual Return) / (Maximum Drawdown in absolute value).
    Requires the Maximum Drawdown to be already calculated.
    data: DataFrame with 'daily_returns' and 'close' (to calculate max drawdown).
    """
    # Average annual return (simple) = daily average * 252
    annual_return = data['daily_returns'].mean() * 252

    # Calculate the max drawdown
    cumulative_max = data['close'].cummax()
    drawdown = (data['close'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # If max_drawdown is zero or close to zero, avoid division by zero
    if max_drawdown == 0:
        return np.inf

    calmar_ratio = annual_return / abs(max_drawdown)
    return calmar_ratio



###############################################################################
#                           4) Beta and Alpha + Treynor                        #
###############################################################################
def calculate_beta_alpha(data, benchmark_returns, risk_free_rate=0.01):
    # Converts the risk-free rate to daily
    daily_rf = risk_free_rate / 252.0

    # Calculates excess returns
    portfolio_excess = data['daily_returns'] - daily_rf
    market_excess = benchmark_returns - daily_rf

    # Aligns the time series
    portfolio_excess, market_excess = portfolio_excess.align(market_excess, join='inner')

    # Removes duplicates
    portfolio_excess = portfolio_excess[~portfolio_excess.index.duplicated(keep='first')]
    market_excess = market_excess[~market_excess.index.duplicated(keep='first')]

    # Removes missing data
    portfolio_excess = portfolio_excess.dropna()
    market_excess = market_excess.dropna()

    # Compares indices
    difference = portfolio_excess.index.symmetric_difference(market_excess.index)
    if not difference.empty:
        print(f"Differences found in indices: {difference}")

    # Trims to the minimum length, if needed
    min_length = min(len(portfolio_excess), len(market_excess))
    portfolio_excess = portfolio_excess.iloc[:min_length]
    market_excess = market_excess.iloc[:min_length]

    # Final size check
    if len(portfolio_excess) != len(market_excess):
        raise ValueError(f"Persistent error: Incompatible sizes between portfolio_excess ({len(portfolio_excess)}) and market_excess ({len(market_excess)}).")

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(market_excess, portfolio_excess)

    # Annualized Beta and Alpha
    beta_annual = slope
    alpha_annual = intercept * 252

    return beta_annual, alpha_annual


def calculate_treynor_ratio(data, benchmark_data, risk_free_rate=0.01):
    """
    Calculates the Treynor Ratio = (Rp - Rf) / Beta
    data: DataFrame with 'daily_returns'.
    benchmark_data: array-like with the benchmark returns.
    risk_free_rate: annual risk-free rate (e.g., 0.01 -> 1%).
    """
    beta, _ = calculate_beta_alpha(data, benchmark_data, risk_free_rate=risk_free_rate)

    # Annual portfolio return
    rp_annual = data['daily_returns'].mean() * 252
    # Annual risk-free return
    rf_annual = risk_free_rate

    if beta == 0:
        return np.inf

    treynor_ratio = (rp_annual - rf_annual) / beta
    return treynor_ratio

###############################################################################
#                      5) Historical VaR (Non-Parametric)                      #
###############################################################################
def calculate_historical_var(data, confidence_level=0.95):
    """
    Calculates the historical VaR (non-parametric).
    confidence_level: e.g., 0.95 for 95%.
    """
    returns = data['daily_returns'].dropna().values
    # Sort from lowest (worst) to highest (best)
    sorted_returns = np.sort(returns)

    # Corresponding percentile (if 95%, cutoff = 5%)
    cutoff_index = int((1 - confidence_level) * len(sorted_returns))
    if cutoff_index < 0:
        cutoff_index = 0
    var_historical = sorted_returns[cutoff_index]

    return var_historical




