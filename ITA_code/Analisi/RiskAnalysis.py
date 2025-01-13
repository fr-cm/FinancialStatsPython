import numpy as np
from scipy.stats import linregress, norm
import logging
import pandas as pd


########################################################################################################
########################################################################################################
#                                            volatility                                                #
########################################################################################################
########################################################################################################
def calculate_volatility(data):
    """
    Calcola la volatilità storica (deviazione standard dei rendimenti) annualizzata.
    """
    volatility = data['daily_returns'].std() * np.sqrt(252) * 100  # Annualizzata e in percentuale
    return volatility



########################################################################################################
########################################################################################################
#                                            rsi                                                       #
########################################################################################################
########################################################################################################

def calculate_rsi(data, period=14):
    """
    Calcola il Relative Strength Index (RSI) senza sovrascrivere colonne esistenti.
    """
    if 'rsi' in data.columns:
        print("Avviso: La colonna 'rsi' esiste già. Salto l'aggiunta.")
        return data
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

# Configura il logging
logging.basicConfig(
    filename='../../analisi.log',  # File di log
    filemode='a',  # Modalità di append
    level=logging.DEBUG,  # Livello di logging
    format='%(asctime)s - %(levelname)s - %(message)s'
)



########################################################################################################
########################################################################################################
#                                            calculate bollinger bands                                 #
########################################################################################################
########################################################################################################
def calculate_bollinger_bands(data, period=20):
    """
    Calcola le bande di Bollinger senza sovrascrivere colonne esistenti.
    """
    if 'bollinger_upper' in data.columns or 'bollinger_lower' in data.columns:
        print("Avviso: Le colonne 'bollinger_upper' o 'bollinger_lower' esistono già. Salto l'aggiunta.")
        return data
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    data['bollinger_upper'] = sma + (std * 2)
    data['bollinger_lower'] = sma - (std * 2)
    return data



########################################################################################################
########################################################################################################
#                                            max drawdown                                              #
########################################################################################################
########################################################################################################
def calculate_max_drawdown(data):
    """
    Calcola il massimo drawdown.
    """
    cumulative_max = data['close'].cummax()
    drawdown = (data['close'] - cumulative_max) / cumulative_max
    return drawdown.min().item()



########################################################################################################
########################################################################################################
#                                            sharp ratio                                               #
########################################################################################################
########################################################################################################
def calculate_sharpe_ratio(data, risk_free_rate=0.01):
    """
    Calcola l'indice di Sharpe.
    risk_free_rate: Tasso privo di rischio annualizzato.
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
    Calcola il Value at Risk (VaR) al 95%.
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
    Calcola l'Expected Shortfall (o CVaR) con approccio storico.
    confidence_level: livello di confidenza, es. 0.95 per 95%.
    float, stima dell'Expected Shortfall (CVaR) come valore (rendimento).
             Se vuoi un numero positivo che esprima la 'perdita', puoi prendere il valore assoluto.
    """
    returns = data['daily_returns'].dropna().values
    # Ordina i rendimenti dal più basso al più alto
    sorted_returns = np.sort(returns)

    # Indice corrispondente al cutoff (5% se confidence_level=95%)
    cutoff_index = int((1 - confidence_level) * len(sorted_returns))
    if cutoff_index < 1:
        cutoff_index = 1  # nel caso ci siano pochi dati

    # VaR storico (al percentile 5% se confidence_level=95%)
    var_historical = sorted_returns[cutoff_index]

    # Seleziona tutti i rendimenti peggiori o uguali al VaR
    tail_losses = sorted_returns[:cutoff_index]

    # Se per caso non ci sono abbastanza dati in coda, evita il div / 0
    if len(tail_losses) == 0:
        return var_historical

    # Expected Shortfall = media dei rendimenti nella coda (peggiori)
    cvar = tail_losses.mean()

    return cvar


###############################################################################
#                            2) Sortino Ratio                                  #
###############################################################################
def calculate_sortino_ratio(data, risk_free_rate=0.01, target_return=0.0):
    """
    Calcola il Sortino Ratio.
    risk_free_rate: Tasso risk-free annuale (es: 0.01 -> 1%).
    target_return: Rendimento target (spesso 0, oppure risk_free).

    """
    # Converte il risk-free rate annuo in giornaliero (se 252 giorni di trading)
    daily_rf = risk_free_rate / 252.0

    # Rendimento in eccesso rispetto al target (o risk-free)
    excess_returns = data['daily_returns'] - daily_rf - target_return

    # Downside deviation (deviazione standard solo quando excess_return < 0)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        # ATTENZIONE!! Se non ci sono valori negativi, Sortino tende a infinito
        return np.inf

    downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

    # Rendimento medio annuo in eccesso
    mean_excess_annual = (excess_returns.mean() * 252)
    # Downside deviation annualizzata
    downside_deviation_annual = downside_deviation * np.sqrt(252)

    sortino_ratio = mean_excess_annual / downside_deviation_annual
    return sortino_ratio


###############################################################################
#                             3) Calmar Ratio                                  #
###############################################################################
def calculate_calmar_ratio(data):
    """
    Calcola il Calmar Ratio = (Rendimento annuo medio) / (Max Drawdown in valore assoluto).
    Necessita del Max Drawdown già calcolato.
    data: DataFrame con 'daily_returns' e 'close' (per calcolare max drawdown).
    """
    # Rendimento annuo medio (semplice) = media giornaliera * 252
    annual_return = data['daily_returns'].mean() * 252

    # Calcolo del max drawdown
    cumulative_max = data['close'].cummax()
    drawdown = (data['close'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # Se max_drawdown è zero o vicino a zero, evitare la divisione per zero
    if max_drawdown == 0:
        return np.inf

    calmar_ratio = annual_return / abs(max_drawdown)
    return calmar_ratio


###############################################################################
#                           4) Beta e Alpha + Treynor                          #
###############################################################################
def calculate_beta_alpha(data, benchmark_returns, risk_free_rate=0.01):
    # Converte il risk-free in giornaliero
    daily_rf = risk_free_rate / 252.0

    # Calcola rendimenti in eccesso
    portfolio_excess = data['daily_returns'] - daily_rf
    market_excess = benchmark_returns - daily_rf

    # Allineamento delle serie temporali
    portfolio_excess, market_excess = portfolio_excess.align(market_excess, join='inner')

    # Rimozione duplicati
    portfolio_excess = portfolio_excess[~portfolio_excess.index.duplicated(keep='first')]
    market_excess = market_excess[~market_excess.index.duplicated(keep='first')]

    # Rimozione dati mancanti
    portfolio_excess = portfolio_excess.dropna()
    market_excess = market_excess.dropna()

    # Confronto indici
    difference = portfolio_excess.index.symmetric_difference(market_excess.index)
    if not difference.empty:
        print(f"Differenze trovate negli indici: {difference}")

    # Taglia alla lunghezza minima, se necessario
    min_length = min(len(portfolio_excess), len(market_excess))
    portfolio_excess = portfolio_excess.iloc[:min_length]
    market_excess = market_excess.iloc[:min_length]

    # Verifica finale delle dimensioni
    if len(portfolio_excess) != len(market_excess):
        raise ValueError(f"Errore persistente: Dimensioni non compatibili tra portfolio_excess ({len(portfolio_excess)}) e market_excess ({len(market_excess)}).")

    # Regressione lineare
    slope, intercept, r_value, p_value, std_err = linregress(market_excess, portfolio_excess)

    # Beta e alpha annualizzati
    beta_annual = slope
    alpha_annual = intercept * 252

    return beta_annual, alpha_annual




def calculate_treynor_ratio(data, benchmark_data, risk_free_rate=0.01):
    """
    Calcola il Treynor Ratio = (Rp - Rf) / Beta
    data: DataFrame con 'daily_returns'.
    benchmark_data: array-like con i rendimenti del benchmark.
    risk_free_rate: tasso risk-free annuale (es: 0.01 -> 1%).
    """
    beta, _ = calculate_beta_alpha(data, benchmark_data, risk_free_rate=risk_free_rate)

    # Rendimento annuo portafoglio
    rp_annual = data['daily_returns'].mean() * 252
    # Rendimento risk-free annuo
    rf_annual = risk_free_rate

    if beta == 0:
        return np.inf

    treynor_ratio = (rp_annual - rf_annual) / beta
    return treynor_ratio


###############################################################################
#                      5) Historical VaR (non parametrico)                     #
###############################################################################
def calculate_historical_var(data, confidence_level=0.95):
    """
    Calcola il VaR storico (non-parametrico).
    confidence_level: ad es. 0.95 per il 95%.

    """
    returns = data['daily_returns'].dropna().values
    # Ordia dal più basso (peggiore) al più alto (migliore)
    sorted_returns = np.sort(returns)

    # Percentile corrispondente (se 95%, cutoff = 5%)
    cutoff_index = int((1 - confidence_level) * len(sorted_returns))
    if cutoff_index < 0:
        cutoff_index = 0
    var_historical = sorted_returns[cutoff_index]

    return var_historical



