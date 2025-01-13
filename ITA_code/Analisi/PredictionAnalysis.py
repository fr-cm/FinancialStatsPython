import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


########################################################################################################
#                                     1. Check stationarity                                            #
########################################################################################################
def check_stationarity(data, column='close'):
    """
    Verifica la stazionarietà della serie temporale utilizzando il test di Dickey-Fuller.
    Restituisce i risultati del test Dickey-Fuller (test statistic, p-value, etc.).

    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Errore: 'data' non è un DataFrame.")
    if column not in data.columns:
        raise ValueError(f"Errore: Il DataFrame non contiene la colonna '{column}'.")

    serie = data[column].dropna()
    if serie.empty:
        raise ValueError(f"Errore: La colonna '{column}' è vuota dopo aver rimosso i valori NaN.")

    # Esegui il test di Dickey-Fuller
    result = adfuller(serie)
    return result


########################################################################################################
#                                     2. Make series stationary                                        #
########################################################################################################
def make_stationary(data, column='close'):
    """
    Applica la differenziazione di primo ordine alla serie per renderla (potenzialmente) stazionaria
    senza sovrascrivere la colonna esistente.
    """
    diff_column = f'diff_{column}'
    if diff_column in data.columns:
        print(f"Avviso: La colonna '{diff_column}' esiste già. Salto l'aggiunta.")
        return data

    data[diff_column] = data[column].diff()

    return data


########################################################################################################
#                                     3. Selezione automatica ARIMA                                    #
########################################################################################################
def select_arima_model(data, max_p=5, max_d=2, max_q=5):
    """
    Esegue una grid search su p, d, q in [0..max_p], [0..max_d], [0..max_q]
    per trovare il modello ARIMA con AIC minimo.

    :param data: Serie temporale (pandas Series) da modellare.
    :param max_p: Valore massimo per p (parte AR).
    :param max_d: Valore massimo per d (ordine di differenziazione).
    :param max_q: Valore massimo per q (parte MA).
    :return: Modello ARIMA fittato con il miglior AIC e la tupla (p, d, q).
    """
    warnings.filterwarnings("ignore")  # Ignora i warning per semplificare l'output

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
        print(f"\nMiglior ARIMA{best_order} con AIC={best_aic:.2f}")
        return best_model, best_order
    else:
        print("Nessun modello ARIMA valido trovato.")
        return None, None


########################################################################################################
#                                     4. Applica un ARIMA specifico                                     #
########################################################################################################
def apply_arima(data, column='close', order=(1, 1, 1), start_params=None):
    """
    Applica il modello ARIMA ai dati di chiusura (o altra colonna), usando un ordine specifico.
    order: Ordine del modello ARIMA (p, d, q).
    start_params: Parametri iniziali, se desiderati.

    """
    serie = data[column].dropna()
    model = ARIMA(serie, order=order)

    if start_params:
        model_fit = model.fit(start_params=start_params)
    else:
        model_fit = model.fit()

    return model_fit


########################################################################################################
#                                     5. Applica modello GARCH (1,1)                                   #
########################################################################################################
def apply_garch(data, returns_col='daily_returns'):
    """
    Applica il modello GARCH(1,1) sui rendimenti giornalieri.

    """
    # Scala i rendimenti al 100%
    returns = data[returns_col].dropna() * 100
    garch = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    return garch_fit
