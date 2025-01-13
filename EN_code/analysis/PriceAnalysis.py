import pandas as pd

########################################################################################################
########################################################################################################
#                                            Average Price                                            #
########################################################################################################
########################################################################################################
def calculate_average_price(data):
    """
    Calculates the average price.
    """
    if data.empty or 'close' not in data.columns:
        print("Warning: Empty DataFrame or 'close' column not found.")
        return None
    average_price = data['close'].dropna().mean()
    return average_price


########################################################################################################
########################################################################################################
#                                            Daily Returns                                             #
########################################################################################################
########################################################################################################
def calculate_daily_returns(data, add_column=True):
    """
    Calculates the daily returns without overwriting existing columns.
    """
    # Check that `data` is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Error: 'data' must be a DataFrame. Received type: {type(data)}")

    # Check that `close` exists and `data` is not empty
    if 'close' not in data.columns:
        raise ValueError("The DataFrame must contain a 'close' column.")
    if data.empty:
        raise ValueError("The DataFrame is empty.")
    # Calculate the daily returns
    daily_returns = data['close'].pct_change()
    # Add the 'daily_returns' column to the DataFrame if requested
    if add_column:
        if 'daily_returns' in data.columns:
            print("Warning: The 'daily_returns' column already exists. Skipping addition.")
            return data
        data['daily_returns'] = daily_returns
        return data
    return daily_returns



########################################################################################################
########################################################################################################
#                                            Moving Averages                                           #
########################################################################################################
########################################################################################################
def calculate_moving_averages(data, periods=[50, 200]):
    """
    Adds columns with moving averages to the DataFrame without overwriting existing columns.
    """
    if data is None or 'close' not in data.columns or data.empty:
        raise ValueError("The DataFrame must contain a non-empty 'close' column.")
    for period in periods:
        col_name = f'sma_{period}'
        if col_name in data.columns:
            print(f"Warning: The column '{col_name}' already exists. Skipping addition.")
            continue
        data[col_name] = data['close'].rolling(window=period).mean()
    return data


########################################################################################################
########################################################################################################
#                                            ema                                              #
########################################################################################################
########################################################################################################
def calculate_ema(data, periods=[50, 200]):
    """
    Adds columns with exponential moving averages to the DataFrame without overwriting existing columns.
    """
    if data is None or 'close' not in data.columns or data.empty:
        raise ValueError("The DataFrame must contain a non-empty 'close' column.")
    for period in periods:
        col_name = f'ema_{period}'
        if col_name in data.columns:
            print(f"Warning: The column '{col_name}' already exists. Skipping addition.")
            continue
        data[col_name] = data['close'].ewm(span=period, adjust=False).mean()
    return data

########################################################################################################
########################################################################################################
#                                            Volume                                                  #
########################################################################################################
########################################################################################################
def calculate_average_volume(data, period=50):
    """
    Calculates the average volume over a specific period and adds the column to the DataFrame.
    Returns the entire DataFrame.
    """
    if 'volume' not in data.columns:
        raise ValueError("The DataFrame must contain a 'volume' column.")
    col_name = f'avg_volume_{period}'

    # If the column already exists, avoid recreating or overwriting it
    if col_name in data.columns:
        print(f"Warning: The column '{col_name}' already exists. Skipping addition.")
        return data
    # Calculate rolling mean
    data[col_name] = data['volume'].rolling(window=period).mean()
    return data


########################################################################################################
########################################################################################################
#                                            Momentum                                                #
########################################################################################################
########################################################################################################
def calculate_momentum(data, period=14):
    """
    Calculates the momentum indicator without overwriting existing columns.
    """
    col_name = f'momentum_{period}'
    if 'close' not in data.columns:
        raise ValueError("The DataFrame must contain a 'close' column.")
    if col_name in data.columns:
        print(f"Warning: The column '{col_name}' already exists. Skipping addition.")
        return data
    data[col_name] = data['close'] - data['close'].shift(period)
    return data



