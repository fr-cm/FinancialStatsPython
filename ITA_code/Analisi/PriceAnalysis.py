import pandas as pd

########################################################################################################
########################################################################################################
#                                            prezzo medio                                              #
########################################################################################################
########################################################################################################
def calculate_average_price(data):
    """
    Calcola il prezzo medio.
    """
    if data.empty or 'close' not in data.columns:
        print("Avviso: DataFrame vuoto o colonna 'close' non trovata.")
        return None
    average_price = data['close'].dropna().mean()
    return average_price

########################################################################################################
########################################################################################################
#                                            rendimenti giornalieri                                    #
########################################################################################################
########################################################################################################
def calculate_daily_returns(data, add_column=True):
    """
    Calcola i rendimenti giornalieri senza sovrascrivere colonne esistenti.
    """
    # Controllo `data` sia un DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Errore: 'data' deve essere un DataFrame. Tipo ricevuto: {type(data)}")

    # Controllo `close` esista e che `data` non sia vuoto
    if 'close' not in data.columns:
        raise ValueError("Il DataFrame deve contenere una colonna 'close'.")
    if data.empty:
        raise ValueError("Il DataFrame è vuoto.")
    # Calcolo dei rendimenti giornalieri
    daily_returns = data['close'].pct_change()
    # Aggiunta della colonna 'daily_returns' al DataFrame, se richiesto
    if add_column:
        if 'daily_returns' in data.columns:
            print("Avviso: La colonna 'daily_returns' esiste già. Salto l'aggiunta.")
            return data
        data['daily_returns'] = daily_returns
        return data
    return daily_returns


########################################################################################################
########################################################################################################
#                                            media mobile                                              #
########################################################################################################
########################################################################################################
def calculate_moving_averages(data, periods=[50, 200]):
    """
    Aggiunge colonne con medie mobili al DataFrame senza sovrascrivere colonne esistenti.
    """
    if data is None or 'close' not in data.columns or data.empty:
        raise ValueError("Il DataFrame deve contenere una colonna 'close' non vuota.")
    for period in periods:
        col_name = f'sma_{period}'
        if col_name in data.columns:
            print(f"Avviso: La colonna '{col_name}' esiste già. Salto l'aggiunta.")
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
    Aggiunge colonne con medie mobili esponenziali al DataFrame senza sovrascrivere colonne esistenti.
    """
    if data is None or 'close' not in data.columns or data.empty:
        raise ValueError("Il DataFrame deve contenere una colonna 'close' non vuota.")
    for period in periods:
        col_name = f'ema_{period}'
        if col_name in data.columns:
            print(f"Avviso: La colonna '{col_name}' esiste già. Salto l'aggiunta.")
            continue
        data[col_name] = data['close'].ewm(span=period, adjust=False).mean()
    return data

########################################################################################################
########################################################################################################
#                                            volume                                              #
########################################################################################################
########################################################################################################
def calculate_average_volume(data, period=50):
    """
    volume medio su un periodo specifico e aggiunge la colonna al DataFrame.
    Ritorna l'intero DataFrame.
    """
    if 'volume' not in data.columns:
        raise ValueError("Il DataFrame deve contenere una colonna 'volume'.")
    col_name = f'avg_volume_{period}'

    # Se la colonna esiste già, evita di ricrearla o la sovrascrivi
    if col_name in data.columns:
        print(f"Avviso: La colonna '{col_name}' esiste già. Salto l'aggiunta.")
        return data
    # Calcolo rolling mean
    data[col_name] = data['volume'].rolling(window=period).mean()
    return data

########################################################################################################
########################################################################################################
#                                            momentum                                              #
########################################################################################################
########################################################################################################
def calculate_momentum(data, period=14):
    """
    Calcola l'indicatore di momentum senza sovrascrivere colonne esistenti.
    """
    col_name = f'momentum_{period}'
    if 'close' not in data.columns:
        raise ValueError("Il DataFrame deve contenere una colonna 'close'.")
    if col_name in data.columns:
        print(f"Avviso: La colonna '{col_name}' esiste già. Salto l'aggiunta.")
        return data
    data[col_name] = data['close'] - data['close'].shift(period)
    return data


