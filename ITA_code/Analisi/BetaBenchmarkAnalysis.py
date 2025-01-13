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
    Calcola il beta dell'azione rispetto a un benchmark.
    """
    if benchmark_data is None or benchmark_data.empty:
        print("    : Errore: Benchmark non valido o dati mancanti.")
        return None

    merged = pd.DataFrame({
        'Stock': data['daily_returns'],
        'Benchmark': benchmark_data['daily_returns']
    }).dropna()

    if merged.empty:
        print("Errore: Nessun dato sovrapponibile tra l'azione e il benchmark.")
        return None

    beta, _, _, _, _ = linregress(merged['Benchmark'], merged['Stock'])
    return beta


########################################################################################################
########################################################################################################
#                                            correlation cointegration                                 #
########################################################################################################
########################################################################################################
def calculate_correlation_and_cointegration(stock_data, benchmark_data):
    # Allinea i daily_returns
    returns_stock = stock_data['daily_returns'].dropna()
    returns_bench = benchmark_data['daily_returns'].dropna()
    returns_stock, returns_bench = returns_stock.align(returns_bench, join='inner')

    # Allinea i 'close'
    close_stock = stock_data['close'].dropna()
    close_bench = benchmark_data['close'].dropna()
    close_stock, close_bench = close_stock.align(close_bench, join='inner')

    # Calcola la correlazione sui rendimenti
    correlation = returns_stock.corr(returns_bench)

    # Calcola la cointegrazione sui prezzi
    score, pvalue, _ = coint(close_stock, close_bench)

    return correlation, pvalue