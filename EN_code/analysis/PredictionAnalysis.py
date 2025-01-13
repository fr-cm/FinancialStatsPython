import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


########################################################################################################
#                                     1. Check Stationarity                                           #
########################################################################################################
def check_stationarity(data, column='close'):
    """
    Checks the stationarity of the time series using the Dickey-Fuller test.
    Returns the results of the Dickey-Fuller test (test statistic, p-value, etc.).
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Error: 'data' is not a DataFrame.")
    if column not in data.columns:
        raise ValueError(f"Error: The DataFrame does not contain the '{column}' column.")

    serie = data[column].dropna()
    if serie.empty:
        raise ValueError(f"Error: The '{column}' column is empty after removing NaN values.")

    # Run the Dickey-Fuller test
    result = adfuller(serie)
    return result



########################################################################################################
#                                     2. Make Series Stationary                                        #
########################################################################################################
def make_stationary(data, column='close'):
    """
    Applies first-order differencing to the series to make it (potentially) stationary
    without overwriting the existing column.
    """
    diff_column = f'diff_{column}'
    if diff_column in data.columns:
        print(f"Warning: The column '{diff_column}' already exists. Skipping addition.")
        return data

    data[diff_column] = data[column].diff()

    return data



########################################################################################################
#                                     3. Automatic ARIMA Model Selection                              #
########################################################################################################
def select_arima_model(data, max_p=5, max_d=2, max_q=5):
    """
    Performs a grid search on p, d, q in [0..max_p], [0..max_d], [0..max_q]
    to find the ARIMA model with the lowest AIC.

    :param data: Time series (pandas Series) to be modeled.
    :param max_p: Maximum value for p (AR part).
    :param max_d: Maximum value for d (differencing order).
    :param max_q: Maximum value for q (MA part).
    :return: ARIMA model fitted with the best AIC and the tuple (p, d, q).
    """
    warnings.filterwarnings("ignore")  # Ignore warnings to simplify output

    best_aic = float("inf")
    best_order = None
    best_model = None
    serie = data.dropna()

    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(serie, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except Exception:

                    continue

    if best_model is not None:
        print(f"\nBest ARIMA{best_order} with AIC={best_aic:.2f}")
        return best_model, best_order
    else:
        print("No valid ARIMA model found.")
        return None, None



########################################################################################################
#                                     4. Apply a Specific ARIMA Model                                #
########################################################################################################
def apply_arima(data, column='close', order=(1, 1, 1), start_params=None):
    """
    Applies the ARIMA model to the closing data (or another column), using a specific order.
    order: Order of the ARIMA model (p, d, q).
    start_params: Initial parameters, if desired.
    """
    serie = data[column].dropna()
    model = ARIMA(serie, order=order)

    if start_params:
        model_fit = model.fit(start_params=start_params)
    else:
        model_fit = model.fit()

    return model_fit


########################################################################################################
#                                     5. Apply GARCH(1,1) Model                                      #
########################################################################################################
def apply_garch(data, returns_col='daily_returns'):
    """
    Applies the GARCH(1,1) model to daily returns.
    """
    # Scale the returns by 100%
    returns = data[returns_col].dropna() * 100
    garch = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    return garch_fit

