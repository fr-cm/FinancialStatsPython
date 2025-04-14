######################################################
#se il codice ha problemi a trovare i tiker 
# Aggiorna la libreria yfinance all'avvio dello script
######################################################
#import subprocess
#import sys
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'yfinance'])


from Analisi.DataRecall import get_benchmark_data, spinner, get_default_dates, get_stock_info, get_logo_url, slow_print, \
    get_stock_data,  \
    generate_dashboard, get_valid_date
from Analisi.PriceAnalysis import calculate_daily_returns, calculate_moving_averages, calculate_average_price, \
    calculate_ema, \
    calculate_momentum, calculate_average_volume
from Analisi.RiskAnalysis import calculate_volatility, calculate_rsi, calculate_bollinger_bands, calculate_max_drawdown, \
    calculate_sharpe_ratio, calculate_var, calculate_sortino_ratio, calculate_treynor_ratio, calculate_calmar_ratio, \
    calculate_historical_var, calculate_beta_alpha, calculate_expected_shortfall
from Analisi.BetaBenchmarkAnalysis import calculate_beta, calculate_correlation_and_cointegration
from Analisi.PredictionAnalysis import make_stationary, select_arima_model, apply_garch, check_stationarity
from ITA_code.Grafici.PlotlyChart import plot_var_analysis, plot_residual, plot_seasonality, plot_trend, \
    plot_stationarity, plot_arima_predictions, plot_garch_volatility, plot_momentum, plot_average_volume, \
    plot_moving_averages, plot_rsi_plotly, plot_bollinger_bands_plotly, plot_daily_returns_plotly, \
    perform_seasonal_decomposition_plotly, plot_data_plotly, plot_story_plotly, plot_histogram_plotly, \
    plot_correlation_plotly, plot_all_models_forecast, plot_adfuller_test, plot_seasonality_and_residual
from ITA_code.Grafici.TerminalChart import plot_adfuller_test, plot_rsi_plotext, plot_bollinger_bands_plotext, \
    perform_seasonal_decomposition_plotext
from ITA_code.Analisi.PredictionAI_1 import run_ai_price_forecast, calculate_split_idx, build_forecast_table
import pandas as pd
import plotext as plt
import plotly.io as pio
from datetime import datetime
import logging

# evita messaggi di errore quando importa tensorflow

import os

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = mostra tutti i log, 1 = rimuove i WARNING, 2 = rimuove INFO, 3 = rimuove tutto fuorché ERROR
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
import os

# Prova a disabilitare i plugin oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Nasconde log di livello INFO, WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
import sys
import threading

# Stili stampa per terminale
BOLD = "\033[1m"
END = "\033[0m"

# Validazione download benchmark
def download_and_validate_benchmark(current_benchmark, start_date, end_date):
    try:
        benchmark_data = get_benchmark_data(current_benchmark, start_date, end_date)
        if benchmark_data is None or benchmark_data.empty:
            return None
        elif 'close' not in benchmark_data.columns:
            return None
        else:
            benchmark_data = calculate_daily_returns(benchmark_data)
            return benchmark_data
    except Exception as e:
        print(f"\n{BOLD}ERRORE:{END} durante il download/analisi del benchmark '{current_benchmark}': {e}")
        return None


#
def run_analyses():
    """
    funzione principale di analisi per un titolo e un benchmark con output dashboard.
    """
    global benchmark_data
    BOLD = "\033[1m"
    END = "\033[0m"
    print(f"""{BOLD}{"""******************************************************************************************************************************
***************************************          ANALISI FINANZIARIA BY F.CAM          ***************************************
******************************************************************************************************************************
"""}{END}""")
    slow_print(f"""{BOLD}{"""                               |                                      +---------------------------+
                               |           /\\                         |  DATA  |  VALORE  |  VAR  |
                               |          /  \\      /\\                |--------|----------|-------|
                               |         /    \\    /  \\/              |   ***  |    **    |   *   |
                               |        /      \\  /                   |   *    |    ***   |  **   |
                               |       /        \\/                    |   **   |    *     |  **   |
                               |      /                               |        |          |       |
                               +--------------------------------+     +---------------------------+"""}{END}""")
    print(f"""{BOLD}{"""******************************************************************************************************************************
******************************************************************************************************************************
 IL TICKER È UNA SERIE DI LETTERE UTILIZZATA PER IDENTIFICARE UNIVOCAMENTE LE AZIONI DI UNA SOCIETÀ O UN INDICE DI MERCATO. 
 PER YAHOO FINANCE, DEVI INSERIRE IL TICKER ESATTO TROVATO SUL SITO, VALIDO SIA PER AZIENDE CHE PER BENCHMARK DI MERCATO.
                                       ( una volta inseriti i dati clicca invio )
                                                    Cerca il TICKER:
                                               esempio: XYZ Ltd (TICKER)
                                       https://it.finance.yahoo.com/lookup/?s=
******************************************************************************************************************************
"""}{END}""")

    # variabili di defoult per gli imput
    default_ticker = ""
    default_benchmark = "ACWI"
    default_start_date = None
    default_end_date = None

    # Input del ticker dell'azione
    ticker = input(f"{BOLD} >>> {END} Inserisci il ticker dell'azione (es. AAPL): ").strip()

    # Calcola le date predefinite basate sui dati storici del titolo dato in input
    default_start_date, default_end_date = get_default_dates(ticker)
    # date di defoult se non sono presenti nel titolo
    if default_start_date is None or default_end_date is None:
        default_start_date = "2020-01-01"
        default_end_date = datetime.now().strftime('%Y-%m-%d')

    # Input del ticker del benchmark
    benchmark_input = input(
        f"{BOLD} >>> {END} Inserisci il ticker del benchmark (opzionale, es. ^GSPC, lascia vuoto per usare il default '{default_benchmark}'):  ").strip()

    # Usa default_benchmark
    benchmark = benchmark_input if benchmark_input else default_benchmark

    # Funzione di validazione delle date
    start_date = get_valid_date(
        f"{BOLD} >>> {END} Inserisci la data di inizio (YYYY-MM-DD) [Default: {default_start_date}]:  ",
        default_start_date)
    end_date = get_valid_date(
        f"{BOLD} >>> {END} Inserisci la data di fine (YYYY-MM-DD) [Default: {default_end_date}]:  ",
        default_end_date)

    # Prende il logo dell'azione per la dashboard
    logo_url = get_logo_url(ticker)
    # Scarica i dati per l'azione
    data = get_stock_data(ticker, start_date, end_date)

    # Validazione del risultato di `data`
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError(f"Errore: 'get_stock_data' non ha restituito un DataFrame valido per il ticker {ticker}.")
    if data.empty:
        raise ValueError(f"Errore: Nessun dato disponibile per il ticker {ticker}.")
    if 'close' not in data.columns:
        raise ValueError(f"Errore: Il DataFrame non contiene la colonna 'close'.")

    # Inizializza benchmark_data per evitare errori
    benchmark_data = None

    # funzioneper provare a scaricare il benchmark
    if benchmark:
        benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)
        if benchmark_data is None:
            while True:
                benchmark_input = input(
                    f"{BOLD} >>> {END} Inserisci un ticker valido per il benchmark (opzionale, lascia vuoto per usare il default '{default_benchmark}'):  "
                ).strip()

                # Se l'utente lascia vuoto, usa il default_benchmark
                if not benchmark_input:
                    print(
                        f"{BOLD}INFO:{END} Nessun input fornito. Utilizzo del benchmark predefinito '{default_benchmark}'.")
                    benchmark = default_benchmark
                else:
                    benchmark = benchmark_input

                # Prova a scaricare il benchmark con il nuovo input
                benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)

                if benchmark_data is not None:
                    break
                else:
                    # Se viena lasciato vuoto e ha tentato di usare il default_benchmark, prova a scaricarlo
                    if not benchmark_input:
                        print(
                            f"{BOLD}INFO:{END} Tentativo di utilizzo del benchmark predefinito '{default_benchmark}'.")
                        benchmark = default_benchmark
                        benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)
                        if benchmark_data is not None:
                            break

    #############################################################################
    #     ANALISI
    #############################################################################

    average_price = calculate_average_price(data)
    last_price = data['close'].iloc[-1] if not data['close'].empty else None
    if last_price is not None:
        price_position = "sopra" if last_price > average_price else "sotto"
    else:
        price_position = "N/A"
    data = calculate_daily_returns(data)
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_ema(data, periods=[50, 200])
    data = calculate_momentum(data, period=14)
    data = calculate_average_volume(data, period=50)
    volatility = calculate_volatility(data)
    max_drawdown = calculate_max_drawdown(data)
    sharpe_ratio = calculate_sharpe_ratio(data)
    var_value = calculate_var(data)
    adf_result = check_stationarity(data)  # test Dickey-Fuller
    adf_pvalue = adf_result[1]
    messaggio = None
    #stop_event = None
    stop_event = threading.Event()
    spinner_thread = None

    if adf_pvalue >= 0.05:
        slow_print("La serie non è stazionaria.")
        slow_print("Applicazione della differenziazione.")
        data = make_stationary(data)
        stop_event = threading.Event()

        #############################################################################
        #     inizializzazione animazione caricaremento
        #############################################################################
        messaggio = (f"{BOLD} {' >>> '}{END} Elaborazione ")
        spinner_thread = threading.Thread(target=spinner, args=(messaggio, stop_event,))
        spinner_thread.start()


    else:
        slow_print("La serie è stazionaria.")
        messaggio = (f"{BOLD} {' >>> '}{END} Elaborazione ")
        spinner_thread = threading.Thread(target=spinner, args=(messaggio, stop_event,))
        spinner_thread.start()

    # Test di stazionarietà

    # Calcolo del beta con il benchmark
    beta = None
    correlation = None
    coint_pvalue = None
    if benchmark_data is not None:
        beta = calculate_beta(data, benchmark_data)
        correlation, coint_pvalue = calculate_correlation_and_cointegration(data, benchmark_data)

    # Applica ARIMA con la funzione personalizzata (ricerca auto di p, d, q) invece di :
    arima_model, best_order = select_arima_model(data['close'])

    if arima_model is not None:
        arima_summary_html = ''.join([table.as_html() for table in arima_model.summary().tables])
        print(f"Modello ARIMA migliore: ARIMA{best_order}")
    else:
        arima_summary_html = "ARIMA non disponibile."
        print("Nessun modello ARIMA valido trovato.")

    # GARCH
    garch_result = apply_garch(data)
    # CVaR
    cvar_95 = calculate_expected_shortfall(data, confidence_level=0.95)
    # Sortino
    sortino = calculate_sortino_ratio(data, risk_free_rate=0.01, target_return=0.0)
    # Calmar
    calmar = calculate_calmar_ratio(data)
    # Beta e Alpha + Treynor Ratio
    # Assume di avere un array/Series benchmark_returns con i rendimenti giornalieri di un indice
    if benchmark_data is not None:
        beta, alpha = calculate_beta_alpha(data, benchmark_data['daily_returns'], risk_free_rate=0.01)
        treynor = calculate_treynor_ratio(data, benchmark_data['daily_returns'], risk_free_rate=0.01)
    else:
        beta, alpha, treynor = None, None, None

    # Historical VaR
    var_h = calculate_historical_var(data, confidence_level=0.95)

    # Generazione dei grafici Plotly
    plot_var_analysis_fig = plot_var_analysis(data, var_value, var_h, cvar_95, confidence_level=0.95)
    plot_bollinger_bands_fig = plot_bollinger_bands_plotly(data, ticker)
    plot_data_fig = plot_data_plotly(data, ticker)
    plot_story_fig = plot_story_plotly(data, ticker)
    plot_rsi_fig = plot_rsi_plotly(data, ticker)
    plot_daily_returns_fig = plot_daily_returns_plotly(data, ticker)
    plot_histogram_fig = plot_histogram_plotly(data, ticker)
    perform_seasonal_decomposition_fig = perform_seasonal_decomposition_plotly(data, ticker)
    sma_ema_fig = plot_moving_averages(data, ticker)
    volume_fig = plot_average_volume(data, ticker)
    momentum_fig = plot_momentum(data, ticker)
    plot_stationarity_fig = plot_stationarity(data, column='close', diff_column='diff_close')
    plot_arima_fig = plot_arima_predictions(data, arima_model, column='close')
    plot_adf_fig = plot_adfuller_test(adf_result, ticker)
    plot_garch_fig = plot_garch_volatility(data, garch_result)
    plot_seasonality_and_residual_fig = plot_seasonality_and_residual(data, ticker)
    plot_trend_fig = plot_trend(data, ticker)
    plot_seasonality_fig = plot_seasonality(data, ticker)
    plot_residual_fig = plot_residual(data, ticker)
    correlation_fig = plot_correlation_plotly(data, benchmark_data, ticker, benchmark)
    plot_adfuller_test_fig = plot_adfuller_test(adf_result, ticker)

    # Conversione dei grafici in HTML
    plot_correlation_plotly_html = pio.to_html(correlation_fig, full_html=False,
                                               include_plotlyjs='cdn') if correlation_fig else ""
    plot_seasonality_and_residual_fig_html = pio.to_html(plot_seasonality_and_residual_fig, full_html=False,
                                                         include_plotlyjs='cdn')
    decomposition_trend_html = pio.to_html(plot_trend_fig, full_html=False, include_plotlyjs='cdn')
    decomposition_seasonality_html = pio.to_html(plot_seasonality_fig, full_html=False, include_plotlyjs='cdn')
    plot_var_analysis_html = pio.to_html(plot_var_analysis_fig, full_html=False, include_plotlyjs='cdn')
    decomposition_residual_html = pio.to_html(plot_residual_fig, full_html=False, include_plotlyjs='cdn')
    plot_bollinger_bands_plotly_html = pio.to_html(plot_bollinger_bands_fig, full_html=False, include_plotlyjs='cdn')
    plot_data_plotly_html = pio.to_html(plot_data_fig, full_html=False, include_plotlyjs='cdn')
    plot_story_plotly_html = pio.to_html(plot_story_fig, full_html=False, include_plotlyjs='cdn')
    plot_rsi_plotly_html = pio.to_html(plot_rsi_fig, full_html=False, include_plotlyjs='cdn')
    plot_daily_returns_plotly_html = pio.to_html(plot_daily_returns_fig, full_html=False, include_plotlyjs='cdn')
    plot_histogram_plotly_html = pio.to_html(plot_histogram_fig, full_html=False, include_plotlyjs='cdn')
    plot_seasonal_decomposition_plotly_html = pio.to_html(perform_seasonal_decomposition_fig, full_html=False,
                                                          include_plotlyjs='cdn')
    sma_ema_html = pio.to_html(sma_ema_fig, full_html=False, include_plotlyjs='cdn')
    volume_html = pio.to_html(volume_fig, full_html=False, include_plotlyjs='cdn')
    momentum_html = pio.to_html(momentum_fig, full_html=False, include_plotlyjs='cdn')
    plot_stationarity_html = pio.to_html(plot_stationarity_fig, full_html=False, include_plotlyjs='cdn')
    plot_arima_html = pio.to_html(plot_arima_fig, full_html=False, include_plotlyjs='cdn')
    plot_garch_html = pio.to_html(plot_garch_fig, full_html=False, include_plotlyjs='cdn')
    plot_adfuller_test_html = pio.to_html(plot_garch_fig, full_html=False, include_plotlyjs='cdn')
    garch_summary_html = ''.join([table.as_html() for table in garch_result.summary().tables])
    plot_adf_html = pio.to_html(plot_adf_fig, full_html=False, include_plotlyjs='cdn')

    # Prende le info generali del titolo e le sistema in tabella
    general_info = get_stock_info(ticker)
    if not general_info:
        print("Errore: Impossibile recuperare le informazioni generali del titolo.")
        return
    analysis_results = {
        'general_info': general_info,
    }
    general_info_no_desc = general_info.copy()
    general_info_no_desc.pop('description', None)
    df = pd.DataFrame([general_info_no_desc])
    general_info_no_desc['Prezzo Medio'] = average_price
    general_info_no_desc['Ultimo Prezzo'] = last_price
    df = pd.DataFrame.from_dict(general_info_no_desc, orient='index', )
    general_info_table = df.to_html(header=False, )

    # Preparazione dei dati da restituire
    analysis_results = {
        'general_info': general_info,
        'average_price': average_price,
        'last_price': last_price,
        'price_position': price_position,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'beta': beta,
        'correlation': correlation,
        'coint_pvalue': coint_pvalue,
        'arima_summary_html': arima_summary_html,
        'garch_summary_html': garch_summary_html,
        'var_value': var_value,
        'adf_pvalue': adf_pvalue,
        'ema_50': data['ema_50'].iloc[-1] if 'ema_50' in data.columns else None,
        'ema_200': data['ema_200'].iloc[-1] if 'ema_200' in data.columns else None,
        'momentum': data['momentum_14'].iloc[-1] if 'momentum_14' in data.columns else None,

        # Grafici in formato HTML
        'plot_var_analysis_html': plot_var_analysis_html,
        'plot_bollinger_bands_html': plot_bollinger_bands_plotly_html,
        'plot_data_html': plot_data_plotly_html,
        'plot_rsi_html': plot_rsi_plotly_html,
        'plot_daily_returns_html': plot_daily_returns_plotly_html,
        'plot_histogram_html': plot_histogram_plotly_html,
        'plot_seasonal_decomposition_html': plot_seasonal_decomposition_plotly_html,
        'plot_adf_html': plot_adf_html,
        'sma_ema_html': sma_ema_html,
        'volume_html': volume_html,
        'momentum_html': momentum_html,
        'plot_stationarity_html': plot_stationarity_html,
        'plot_arima_html': plot_arima_html,
        'plot_garch_html': plot_garch_html,
        'plot_seasonality_and_residual_fig_html': plot_seasonality_and_residual_fig_html,
        'decomposition_trend_html': decomposition_trend_html,
        'decomposition_seasonalty_html': decomposition_seasonality_html,
        'decomposition_residual_html': decomposition_residual_html,

    }
    if 'volume' in data.columns:
        data = calculate_average_volume(data, period=50)  # data rimane DataFrame
        analysis_results['average_volume'] = data['avg_volume_50'].iloc[-1]

        # fine animazione di caricamento
        if stop_event is not None and spinner_thread is not None:
            stop_event.set()
            spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(messaggio) + 2) + '\r')
        sys.stdout.flush()

    ##################################################################
    # OUTPUT CON SPIEGAZIONI
    ##################################################################

    print("")
    print("")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{f'                                                     RISULTATI ANALISI PER {ticker}\n                                                     ({start_date} -> {end_date})   '}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print("")
    print(f"Nome: {general_info['longName']}")
    print(f"Settore: {general_info['sector']}")
    print(f"Industria: {general_info['industry']}")
    print(f"Capitalizzazione di mercato: {general_info['marketCap']}")
    print(f"Rendimento del dividendo: {general_info['dividendYield']}")
    print(f"Beta: {general_info['beta']}")
    print(f"Dipendenti: {general_info['fullTimeEmployees']}")
    print(f"Paese: {general_info['country']}")
    print(f"Sito Web: {general_info['website']}")
    print("")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    slow_print(general_info['description'])
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print("")
    print("")
    print("")

    decomposition = perform_seasonal_decomposition_plotext(data, ticker)
    print("")
    print("")
    print("")

    # 1) Prezzo Medio
    print(f"1) Prezzo Medio: {BOLD}{average_price:.2f}{END} €")
    print(" ")
    slow_print("   - Rappresenta il prezzo di chiusura medio su tutto il periodo di analisi.")
    slow_print(
        "   - Un prezzo medio più alto indica una tendenza al rialzo, mentre uno più basso potrebbe indicare una tendenza al ribasso.")
    slow_print("   - **Range di riferimento:**")
    slow_print("     * > 150 €: Elevato, possibile sopravvalutazione")
    slow_print("     * 100 - 150 €: Moderato, stabile")
    slow_print("     * < 100 €: Basso, potenziale opportunità di acquisto")
    slow_print(
        "   - Utile per confrontare il prezzo attuale con la media storica e valutare la performance relativa.\n")
    slow_print(" ")

    # 2) Ultimo Prezzo
    if last_price is not None:
        print(f"2) Ultimo Prezzo: {BOLD}{last_price:.2f}{END} € ({price_position} rispetto al prezzo medio)")
        print(" ")
        slow_print("   - Confronta l'ultimo prezzo di chiusura disponibile con la media delle chiusure.")
        slow_print("   - Se l'ultimo prezzo è superiore alla media, potrebbe indicare una tendenza positiva recente.")
        slow_print("   - Se inferiore, potrebbe suggerire una pressione al ribasso o una correzione del mercato.")
        slow_print("   - **Interpretazione del posizionamento rispetto alla media:**")
        slow_print("     * > 0: Sopra la media, segnale positivo")
        slow_print("     * < 0: Sotto la media, segnale negativo")
        slow_print(" ")
    else:
        print("2) Ultimo Prezzo non disponibile.")
        print("   - Potrebbe essere dovuto a dati mancanti o a un errore nel recupero delle informazioni.\n")
        print(" ")

    # 3) Momentum
    print(f"\n3) Momentum (14 giorni): {BOLD}{analysis_results['momentum']:.2f}{END} €")
    print(" ")
    slow_print("   - Misura la velocità di variazione del prezzo rispetto a un periodo specifico.")
    slow_print("   - Valori positivi indicano una tendenza al rialzo.")
    slow_print("   - Valori negativi indicano una tendenza al ribasso.")
    slow_print("   - **Range di riferimento:**")
    slow_print("     * > 0: Slancio positivo, possibile continuazione del trend.")
    slow_print("     * < 0: Slancio negativo, possibile inversione del trend.")
    slow_print("   - Utile per confermare o anticipare movimenti del prezzo.\n")
    slow_print(" ")

    # 4) Medie Mobili Esponenziali (EMA)
    print(f"\n4) EMA (Medie Mobili Esponenziali):")
    print(f"   EMA 50: {BOLD}{analysis_results['ema_50']:.2f}{END} €")
    print(f"   EMA 200: {BOLD}{analysis_results['ema_200']:.2f}{END} €")
    print(" ")
    slow_print("   - L'EMA dà maggior peso ai dati recenti rispetto alla media mobile semplice (SMA).")
    slow_print("   - Utilizzata per individuare trend di breve e lungo termine.")
    slow_print("   - **Range di riferimento:**")
    slow_print("     * EMA 50 sopra EMA 200: Tendenza rialzista (Golden Cross).")
    slow_print("     * EMA 50 sotto EMA 200: Tendenza ribassista (Death Cross).")
    slow_print("   - Una EMA più vicina al prezzo attuale suggerisce una reattività maggiore alle variazioni.")
    slow_print(" ")

    # 5)   Volume Medio
    if 'average_volume' in analysis_results and analysis_results['average_volume']:
        print(f"\n5) Volume Medio (50 giorni): {BOLD}{analysis_results['average_volume']:.0f}{END}")
        print(" ")
        slow_print("   - Il volume rappresenta il numero di scambi effettuati in un determinato periodo.")
        slow_print("   - Un volume medio elevato indica un maggiore interesse degli investitori.")
        slow_print("   - **Interpretazione:**")
        slow_print("     * Aumenti di volume accompagnati da movimenti di prezzo suggeriscono forza del trend.")
        slow_print("     * Diminuzioni di volume indicano indebolimento del trend o mancanza di interesse.")
        slow_print(
            "   - Utile per valutare la solidità di un movimento di prezzo e la partecipazione degli investitori.\n")
        slow_print(" ")

    # 6) Volatilità Annualizzata

    print(f"\n6) Volatilità Annualizzata: {BOLD}{volatility:.2f}{END}%")
    print(" ")
    slow_print("   - Misura la dispersione dei rendimenti giornalieri annualizzati.")
    slow_print("   - **Range di riferimento:**")
    slow_print("     * > 30%: Elevata volatilità, alto rischio")
    slow_print("     * 15% - 30%: Moderata volatilità, rischio bilanciato")
    slow_print("     * < 15%: Bassa volatilità, basso rischio")
    slow_print(
        "   - Una volatilità elevata indica maggior rischio e potenziale di oscillazioni significative dei prezzi.")
    slow_print("   - Una volatilità bassa suggerisce un mercato più stabile.\n")
    slow_print(" ")

    # 7) Massimo Drawdown
    print(f"\n7) Massimo Drawdown: {BOLD}{max_drawdown:.2%}{END}")
    print(" ")
    slow_print("   - Rappresenta la massima perdita percentuale dal picco più alto al punto più basso nel periodo.")
    slow_print("   - **Range di riferimento:**")
    slow_print("     * > 50%: Alto rischio di perdita significativa")
    slow_print("     * 20% - 50%: Rischio moderato")
    slow_print("     * < 20%: Basso rischio di perdita")
    slow_print("   - Indica il rischio di perdita massima potenziale durante l'investimento.")
    slow_print("   - Utile per valutare la resilienza del titolo durante periodi di mercato avversi.\n")
    print(" ")

    # 8) Value at Risk (VaR)
    print(f"\n8) Value at Risk (95%): {BOLD}{-var_value:.2%}{END}")
    print(" ")
    slow_print("    - Stima la perdita massima attesa (in percentuale) con un livello di confidenza del 95%.")
    slow_print("    - **Interpretazione del VaR:**")
    slow_print(
        "      * Un VaR del 5% indica che c'è solo il 5% di probabilità che la perdita superi questo valore in un dato periodo.")
    slow_print(
        "    - Utile per valutare il rischio potenziale di un investimento e per implementare strategie di gestione del rischio.\n")
    print(" ")

    # 9) Historical VaR (95%)
    print(f"\n9) Historical VaR (95%): {BOLD}{var_h:.4f}{END}")
    print(" ")
    slow_print("   - VaR indica la peggiore perdita attesa (o soglia di perdita) a un certo livello di confidenza.")
    slow_print("   - Questa versione 'storica' non assume alcuna distribuzione (es. normale).")
    slow_print("   - Si basa semplicemente sulle peggiori performance passate.\n")

    # 10) Expected Shortfall / CVaR (95%)
    print(f"\n10) Expected Shortfall (CVaR) 95%: {BOLD}{cvar_95:.4f}{END}")
    print(" ")
    slow_print("   - Il CVaR indica la perdita media (attesa) in caso di superamento del VaR.")
    slow_print(
        "   - Confronta il VaR: mentre il VaR ti dice la 'soglia', il CVaR ti dice 'quanto si perde in media' oltre quella soglia.")
    slow_print("   - **Valori negativi** → Se interpretati come rendimenti, corrispondono a perdite.")
    slow_print("   - Più il CVaR è grande (in valore assoluto negativo), più pesante è la coda di perdita.\n")

    # 11) Indice di Sharpe
    print(f"\n11) Indice di Sharpe: {BOLD}{sharpe_ratio:.2f}{END}")
    print(" ")
    slow_print("   - Rapporto rischio/rendimento rispetto a un tasso privo di rischio.")
    slow_print("   - **Interpretazione del Sharpe Ratio:**")
    slow_print("     * > 1: Buona performance aggiustata per il rischio")
    slow_print("     * 0.5 - 1: Accettabile performance")
    slow_print("     * < 0.5: Scarsa performance aggiustata per il rischio")
    slow_print("   - Un Sharpe ratio più alto indica una migliore performance aggiustata per il rischio.")
    slow_print(
        "   - Aiuta a confrontare l'efficienza di diversi investimenti in termini di rendimento per unità di rischio.\n")
    print(" ")

    # 12) Treynor Ratio
    print(f"\n12) Treynor Ratio: {BOLD}{treynor:.2f}{END}")
    slow_print("   - Treynor = (Rendimento del portafoglio - Rischio-free) / Beta")
    slow_print("   - A differenza dello Sharpe Ratio, considera solo il rischio sistematico (Beta).")
    slow_print("   - Maggiore è il Treynor, migliore è la remunerazione del rischio sistematico.\n")

    # 13) Sortino Ratio
    print(f"\n13) Sortino Ratio: {BOLD}{sortino:.2f}{END}")
    slow_print("   - Il Sortino Ratio è simile allo Sharpe, ma considera solo la volatilità negativa (downside risk).")
    slow_print("   - Più è alto, meglio il portafoglio remunera il rischio 'di perdita'.")
    slow_print("   - **Range indicativo:**")
    slow_print("     * Sortino > 2: Eccellente")
    slow_print("     * 1 < Sortino < 2: Accettabile")
    slow_print("     * < 1: Da valutare con attenzione\n")

    # 14) Calmar Ratio
    print(f"\n14) Calmar Ratio: {BOLD}{calmar:.2f}{END}")
    slow_print("   - Calmar = (Rendimento annuo medio) / (Max Drawdown in valore assoluto).")
    slow_print("   - Indica quanto 'rendimento annuo' si ottiene per ogni punto % di drawdown.")
    slow_print("   - Più è alto, migliore è il rapporto tra guadagno e rischio di perdita.\n")

    # 15) Beta e Alpha
    print(f"\n15) Beta: {BOLD}{beta:.2f}{END}, Alpha (annuo): {BOLD}{alpha:.4f}{END}")
    slow_print("   - **Beta** misura la sensibilità del tuo asset/portafoglio rispetto al mercato (benchmark).")
    slow_print("     * Beta = 1: Movimento in linea con il mercato")
    slow_print("     * Beta > 1: Amplifica i movimenti di mercato (più rischioso)")
    slow_print("     * Beta < 1: Meno sensibile alle oscillazioni di mercato (meno rischioso)")
    slow_print("   - **Alpha** misura il rendimento in eccesso (o deficit) non spiegato dal Beta.")
    slow_print("     * Alpha positivo = sovra-performance rispetto ai rischi di mercato")
    slow_print("     * Alpha negativo = sotto-performance\n")

    # 16) Beta rispetto al benchmark
    if beta is not None:
        print(f"\n16) Beta rispetto al benchmark ({benchmark}): {BOLD}{beta:.2f}{END}")
        print(" ")
        slow_print("   - Indica quanto il titolo si muove in relazione al benchmark di riferimento.")
        slow_print("   - **Interpretazione del Beta:**")
        slow_print(
            "     * > 1: Il titolo è più volatile del benchmark, indicando maggior rischio e potenziale di rendimento.")
        slow_print("     * < 1: Il titolo è meno volatile del benchmark, suggerendo minore rischio.")
        slow_print("     * = 1: Il titolo si muove in linea con il benchmark.")
        print(" ")
    else:
        slow_print("\n6) Beta: Non disponibile (benchmark non inserito o non valido).")
        slow_print("   - Assicurati di aver fornito un benchmark valido per calcolare il Beta.\n")
        print(" ")

    # 17) Correlazione e Cointegrazione con il benchmark
    if correlation is not None and coint_pvalue is not None:
        print(f"\n17) Correlazione con il benchmark: {BOLD}{correlation:.4f}{END}")
        print(" ")
        slow_print("   - Indica il grado di relazione lineare tra i rendimenti del titolo e del benchmark.")
        print(" ")
        if correlation > 0:
            slow_print("   - Correlazione positiva: i rendimenti tendono a muoversi nella stessa direzione.")
            print(" ")
        else:
            slow_print("   - Correlazione negativa: i rendimenti tendono a muoversi in direzioni opposte.")
        print(f"   P-value Cointegrazione: {BOLD}{coint_pvalue:.4f}{END}")
        print(" ")
        if coint_pvalue < 0.05:
            slow_print("   - Valore < 0.05 indica cointegrazione (relazione di lungo periodo) significativa.")
            slow_print(
                "   - Significa che esiste una relazione stabile tra il titolo e il benchmark nel lungo periodo.")
            print(" ")
        else:
            slow_print("   - Valore >= 0.05 indica che non c'è evidenza sufficiente di cointegrazione significativa.")
            print(" ")
    else:
        slow_print("\n7) Correlazione e Cointegrazione: Non disponibili (benchmark non inserito o non valido).")
        slow_print(
            "   - Verifica di aver fornito dati validi per il benchmark e che il calcolo sia andato a buon fine.\n")
        print(" ")

    # 18) Test Dickey-Fuller
    print(f"\n18) Test Dickey-Fuller (p-value): {BOLD}{adf_pvalue:.4f}{END}")
    print(" ")
    slow_print("    - Verifica la stazionarietà della serie temporale.")
    slow_print("    - **Interpretazione del p-value:**")
    slow_print("      * < 0.05: La serie è stazionaria (ha una media e una varianza costanti nel tempo).")
    slow_print("      * >= 0.05: La serie non è stazionaria.")
    slow_print(
        "    - Serie stazionarie sono una premessa importante per molti modelli di previsione, inclusi ARIMA e GARCH.")
    slow_print("    - Se la serie non è stazionaria, potrebbe essere necessario differenziare o trasformare i dati.\n")
    print(" ")

    # 19) Sommario modello ARIMA
    print(f"\n19) ARIMA(1,1,1) - Sommario modello:\n{arima_model.summary()}")
    print(" ")
    slow_print("   - Il modello ARIMA combina autoregressione (AR) e media mobile (MA) per prevedere i prezzi futuri.")
    slow_print("   - **Componenti chiave nel sommario:**")
    slow_print("     * **Coefficients:** Indicano l'impatto delle componenti AR e MA sul modello.")
    slow_print(
        "     * **p-values:** Valutano la significatività statistica di ciascun coefficiente (valori < 0.05 sono generalmente considerati significativi).")
    slow_print(
        "     * **AIC e BIC:** Criteri di informazione per confrontare modelli; valori più bassi indicano modelli migliori.")
    slow_print("   - Utilizza questo modello per fare previsioni sui futuri prezzi basandoti sui dati storici.\n")
    print(" ")

    # 20) Sommario modello GARCH
    print(f"\n20) GARCH(1,1) - Sommario modello:\n{garch_result.summary()}")
    print(" ")
    slow_print("   - Il modello GARCH stima e prevede la volatilità dei rendimenti.")
    slow_print("   - **Componenti chiave nel sommario:**")
    slow_print("     * **Omega:** Costante nel modello di volatilità.")
    slow_print("     * **Alpha:** Impatto degli shock di mercato passati sulla volatilità attuale.")
    slow_print("     * **Beta:** Influenza della volatilità passata sulla volatilità attuale.")
    slow_print(
        "     * **p-values:** Valutano la significatività dei parametri (valori < 0.05 indicano significatività).")
    slow_print(
        "   - Un modello GARCH ben calibrato aiuta a comprendere e prevedere la volatilità futura, utile nella gestione del rischio.\n")
    print(" ")

    # funzioni grafici terminale
    ticker = f'{ticker}'
    plot_rsi_plotext(data, ticker)
    print(" ")
    print(" ")
    plot_bollinger_bands_plotext(data, ticker)
    print(" ")
    print(" ")
    plt.show()
    slow_print("\nLegenda:")
    print(" - Cyan: Prezzo di Chiusura")
    print(" - Magenta: Bollinger Upper ")
    print(" - Green: Bollinger Lower")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")

    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(" ")

    ####################################################
    # FUNZIONE PER RICHIAMARE O NO L'ADDESTRAMENTO AI
    ####################################################

    use_ai = input(
        f"{BOLD}{'Vuoi eseguire anche la previsione del prezzo con l\'Intelligenza Artificiale? (SI/NO): '}{END}"
    ).strip().lower()

    plot_all_models_forecast_html = "<p><b>Analisi AI non eseguita.</b></p>"
    forecast_html = "<p></p>"

    if use_ai in ['s', 'si', 'y', 'yes']:

        # Esegui la funzione che allena TUTTI i modelli (LSTM, CNN+LSTM, Transformer)
        # e restituisce un dizionario "results" con i dati di test e di forecast per ogni modello
        ai_results = run_ai_price_forecast(
            data=data,
            arima_model=arima_model,
            garch_result=garch_result,
            benchmark_data=benchmark_data,
            future_steps=15  # numero di giorni di previsione
        )

        # Verifica se la funzione ha restituito un risultato valido
        if ai_results is not None and isinstance(ai_results, dict) and len(ai_results) > 0:
            # ai_results è un dizionario di forma:
            # {
            #   "LSTM": {
            #       "best_model": ...,
            #       "forecast_df": pd.DataFrame,
            #       "test_pred_rescaled": np.array,
            #       "y_test_rescaled": np.array,
            #       ...
            #   },
            #   "CNN+LSTM": {...},
            #   "Transformer": {...}
            # }

            # Calcola lo split_idx per allineare la parte di test
            split_idx = calculate_split_idx(data, test_size=0.2)
            # Genera il grafico con la nuova funzione plot_all_models_forecast
            # (questa funzione prende in ingresso TUTTE le predizioni)
            plot_all_models_forecast_fig = plot_all_models_forecast(
                data=data,
                results=ai_results,  # Passiamo il dizionario completo
                split_idx=split_idx
            )

            # Converte il grafico in HTML
            plot_all_models_forecast_html = pio.to_html(
                plot_all_models_forecast_fig,
                full_html=False,
                include_plotlyjs='cdn'
            )

            # concatenando i DataFrame di forecast
            all_forecast_tables_html = ""
            merged_df, table_html = build_forecast_table(ai_results)

            if merged_df is not None:
                all_forecast_tables_html = (table_html.replace('\n', '')
                )
            else:
                all_forecast_tables_html = table_html  # Messaggio di errore/assenza

            forecast_html = all_forecast_tables_html

        else:
            plot_all_models_forecast_html = "<p>Modelli AI non disponibile.</p>"
            forecast_html = "<p>Analisi AI non eseguita.</p>"

    # ASSEGNAZIONE RISULTATI AI
    analysis_results['plot_all_models_forecast_html'] = plot_all_models_forecast_html
    analysis_results['forecast_html'] = forecast_html

    ######################################################
    # IMPALCATURA DASHBOARD
    ######################################################
    data_ds = {
        'navbar': {
            'brand_name': f'{general_info['longName']}',
            'subtitle': f'{ticker}',
            'menu_items': [

                {'id': 'Analisi', 'label': 'Analisi', 'submenu':
                    [
                        {'id': 'prezzo_trend', 'label': 'Prezzo'},
                        {'id': 'Risk', 'label': 'Rischio e Volatilità'},
                        {'id': 'info', 'label': 'info'},
                    ]},
                {'id': 'stagionality', 'label': 'Stagionalità'},
                {'id': 'Modelli', 'label': 'Modelli Previsionali', 'submenu':
                    [
                        {'id': 'ARIMAGARCH', 'label': 'ARIMA e GARCH'},
                        {'id': 'AI', 'label': 'AI'},
                    ]},

            ]
        },
        'pages': [
            {
                'id': 'home',
                'type': 'home',
                'title': f'Analisi {general_info['longName']} <br> Dal {start_date} al {end_date}',
                'image_src': logo_url,
                'contents': [
                    {
                        "type": "home",
                        "id": "home",
                        "content":
                            f"<p>{general_info['description']}</p>"

                    }
                ]
            },
            {
                'id': 'prezzo_trend',
                'type': 'page_menu',
                'title': 'Prezzo e Trend',
                'contents': [
                    {
                        "type": "plot_table_download",
                        "id": "Prezzo",
                        "label": "Prezzo",
                        "plot": plot_data_plotly_html,
                        "title": f"<h3>  </h3>",
                        "data": general_info_table,

                    },
                    {
                        "type": "plot_txt_download",
                        "id": "MediaMobile",
                        "label": "Media Mobile ",
                        "plot": sma_ema_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                            <p>
                            <b>Media Mobile </b><br>
                            La media mobile semplice rappresenta la media aritmetica dei prezzi di chiusura per un determinato periodo. Fornisce una visione lineare del trend del prezzo, riducendo le fluttuazioni a breve termine.<br>
                            - Utilità:<br>
                              * Identificazione di tendenze di lungo e breve termine.<br>
                              * Livelli di supporto e resistenza dinamici.<br>
                            - Interpretazione:<br>
                               Il prezzo sopra la SMA indica una tendenza rialzista.<br>
                               Il prezzo sotto la SMA indica una tendenza ribassista.<br>
                            <br>

                            <b>Media Mobile Esponenziale (EMA)</b><br>
                            EMA 50: <b>{analysis_results['ema_50']:.2f}€</b>  |   EMA 200: <b>{analysis_results['ema_200']:.2f} €</b><br>
                            La media mobile esponenziale dà più peso ai prezzi più recenti rispetto alla SMA, rendendola più reattiva ai cambiamenti di prezzo.<br>
                            
                               Aiuta all'individuazione di tendenze recenti, inversioni e ad analisi di segnali di ingresso e uscita nei mercati finanziari.<br>
                               EMA 50 sopra EMA 200: Segnale rialzista (Golden Cross).<br>
                               EMA 50 sotto EMA 200: Segnale ribassista (Death Cross).<br>
                            - Range di riferimento:<br>
                               EMA 50: Tendenze di breve termine.<br>
                               EMA 200: Tendenze di lungo termine.<br>
                            <br>

                            <p>
                            """,
                        'display': 'none'

                    },
                    {
                        "type": "plot_txt_download",
                        "id": "momentum",
                        "label": "Momentum ",
                        "plot": momentum_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                        <p>
                        <b>Momentum:  {analysis_results['momentum']:.2f}</b><br>
                        Il Momentum misura la velocità con cui il prezzo di un titolo varia rispetto a un periodo specifico. Si calcola confrontando il prezzo corrente con quello di un certo numero di giorni passati.<br>
                        - Formula:<br>
                          <i>Momentum = Prezzo Corrente - Prezzo di N giorni fa</i><br>
                        Aiuta all'identificazione della forza di un trend e alla valutazione delle potenziali inversioni di tendenza.<br>
                        - Interpretazione:<br>
                           <b>Valori positivi:</b> Indicano un trend al rialzo, con il prezzo che si sta rafforzando.<br>
                           <b>Valori negativi:</b> Indicano un trend al ribasso, con il prezzo che sta perdendo forza.<br>
                        - Range di riferimento:<br>
                          * > 0: Slancio positivo, possibile continuazione del trend rialzista.<br>
                          * < 0: Slancio negativo, possibile inversione del trend o indebolimento.<br>
                        <br>
                        </p>
                        """
                        ,
                        'display': 'none'

                    },
                    {
                        "type": "plot_txt_download",
                        "id": "volume",
                        "label": "Volume Medio",
                        "plot": volume_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                        <p>
                        <b>Volume Medio:  {analysis_results['average_volume']}</b><br>
                        Il volume medio rappresenta il numero medio di scambi effettuati su un titolo in un determinato periodo di tempo. Indica il livello di interesse e attività degli investitori sul mercato.<br>
                           Misura l'interesse del mercato verso un titolo.<br>
                           Valutazione della forza di un trend o di un movimento di prezzo.<br>
                        - Interpretazione:<br>
                          Volume elevato: Indica forte interesse da parte degli investitori e spesso conferma la direzione del trend.<br>
                          Volume basso: Può indicare incertezza o mancanza di interesse nel titolo.<br>
                        - Range di riferimento:<br>
                          Aumenti di volume accompagnati da movimenti di prezzo suggeriscono forza del trend.<br>
                          Diminuzioni di volume possono indicare indebolimento del trend o mancanza di partecipazione.<br>
                        <br>
                        </p>
                        """

                        ,
                        'display': 'none'

                    },

                    {
                        "type": "plot_txt",
                        "id": "benchmark_info",
                        "label": "Informazioni Benchmark",
                        "plot": plot_correlation_plotly_html,
                        "title": f"<h2>Analisi Benchmark: {benchmark}</h2>",
                        "text": f"""
                    <p>
                    <b>Beta: {beta}</b><br>
                    Indica quanto il titolo si muove in relazione al benchmark di riferimento.<br>
                    - Interpretazione:<br>
                      * > 1: Più volatile del benchmark<br>
                      * < 1: Meno volatile del benchmark<br>
                      * = 1: In linea con il benchmark<br>
                    <b>Correlazione con il Benchmark: {correlation}</b><br>
                    - Indica il grado di relazione lineare tra i rendimenti del titolo e del benchmark.<br>
                    - Interpretazione:<br>
                      * > 0: Correlazione positiva<br>
                      * < 0: Correlazione negativa<br>
                    <b>P-value Cointegrazione: {coint_pvalue}</b><br>
                    Valuta la presenza di una relazione di lungo periodo tra il titolo e il benchmark.<br>
                    - Interpretazione:<br>
                      * < 0.05: Cointegrazione significativa<br>
                      * >= 0.05: Nessuna evidenza di cointegrazione<br>
                    </p>
                """,
                        'display': 'none'
                    },
                    {
                        "type": "plot",
                        "id": "Rendimenti",
                        "label": "Distribuzione dei Rendimenti",
                        "plot": plot_histogram_plotly_html,
                        'display': 'none'
                    }

                ]
            },
            {
                'id': 'stagionality',
                'type': 'page',
                'title': f'',
                'contents': [
                    {

                        "type": "html",
                        "id": "stagionality4",
                        "label": "Decomposizione Stagionale",
                        "title": f" ",
                        "html": f"""<div style=\" background-color: #e4e4e4;padding-top: 0rem; text-align: center; display:flex; flex-wrap: nowrap; justify-content:centet; flex-direction: column; align-items: center;\"><h2> Decomposizione Stagionale {ticker} dal {start_date} al {end_date}</h2>                       
                                <p>
                                 La decomposizione stagionale scompone la serie temporale in tre componenti:<br>
                                 * <b>Trend:</b> La direzione a lungo termine della serie (ad es., tendenza al rialzo o al ribasso).<br>
                                 * <b>Stagionalità:</b> Fluttuazioni regolari e ripetitive legate a periodi specifici (ad es., stagioni, mesi).<br>
                                 * <b>Residuo:</b> La componente casuale o irregolare restante dopo aver rimosso trend e stagionalità.<br>
                                 Utile per identificare pattern ricorrenti e per migliorare la modellizzazione e le previsioni future.
                                 </p></div>"""
                    },
                    {
                        "type": "plot_plot",
                        "id": "stagionality",
                        "plot1": plot_seasonality_and_residual_fig_html,
                        "plot2": decomposition_trend_html,

                    },
                ]

            },
            {
                'id': 'Risk',
                'type': 'page_menu',
                'title': 'Rischio e Volatilità',
                'contents': [
                    {
                        "type": "plot_txt",
                        "id": "Vol",
                        "label": "Volatilità e Drawdown",
                        "plot": plot_bollinger_bands_plotly_html,
                        "title": "",
                        "text":
                            f"""
                            <p>
                            <b> Volatilità Annualizzata:  {volatility}%</b>  <br>
                            Misura la dispersione dei rendimenti giornalieri annualizzati.<br>
                            - Range di riferimento:<br>
                              * > 30%: Elevata volatilità, alto rischio<br>
                              * 15% - 30%: Moderata volatilità, rischio bilanciato<br>
                              * < 15%: Bassa volatilità, basso rischio<br>
                            - Una volatilità elevata indica maggior rischio e potenziale di oscillazioni significative dei prezzi.
                            - Una volatilità bassa suggerisce un mercato più stabile.<br>
                            <b> Massimo Drawdown: {(max_drawdown * 100)}%</b>  <br>
                            Rappresenta la massima perdita percentuale dal picco più alto al punto più basso nel periodo.<br>
                            - Range di riferimento:<br>
                              * > 50%: Alto rischio di perdita significativa<br>
                              * 20% - 50%: Rischio moderato<br>
                              * < 20%: Basso rischio di perdita<br>
                            Aiuta a valutare la resilienza del titolo durante periodi di mercato avversi.
                            </p>
                            """,
                    },
                    {
                        "type": "plot_txt",
                        "id": "Var",
                        "label": "Value at Risk",
                        "title": "",
                        "plot": plot_var_analysis_html,
                        "text": f"""
                                <p>
                                <b> Value at Risk (95%): {-var_value:.2%}</b> <br>
                               - Stima la perdita massima attesa (in percentuale) con un livello di confidenza del 95%.
                               Un VaR del 5% indica che c'è solo il 5% di probabilità che la perdita superi questo valore in un dato periodo.<br>
                                <b>Historical VaR (95%):{var_h:.4f}</b><br>
                                Il Value at Risk (VaR) storico al 95% rappresenta la perdita massima attesa, con un livello di confidenza del 95%, basata sui dati storici. Non assume alcuna distribuzione statistica, ma utilizza i peggiori rendimenti storici.<br>
                                - Interpretazione:<br>
                                   Un VaR storico del 5% indica che, con il 95% di probabilità, la perdita non supererà questo valore in un dato periodo.<br>
                                   Valori più alti (negativi) indicano maggiore rischio.<br>
                                <b>Expected Shortfall (CVaR) 95%: {cvar_95:.4f}</b><br>
                                L'Expected Shortfall (o CVaR) misura la perdita media attesa nel caso in cui il VaR venga superato. È considerato un indicatore più conservativo rispetto al VaR perché considera l'entità delle perdite nella coda.<br>
                                - Interpretazione:<br>
                                   Il CVaR rappresenta la perdita media in caso di condizioni avverse estreme.<br>
                                   Più il CVaR è negativo, maggiore è il rischio di perdita estrema.<br>
                                </p>""",
                        "display": "none",
                    },
                    {
                        "type": "plot_txt",
                        "id": "INDEX",
                        "label": "Indici",
                        "plot": plot_rsi_plotly_html,
                        "title": "",
                        "text": f"""                                
                         <p>
                                                         <b>RSI (Relative Strength Index)</b><br>
                                L'RSI è un oscillatore che misura la forza e la velocità di un movimento di prezzo recente, fornendo indicazioni su condizioni di ipercomprato o ipervenduto.<br>
                                - Interpretazione:<br>
                                  * < 30: Condizione di ipervenduto, possibile opportunità di acquisto.<br>
                                  * > 70: Condizione di ipercomprato, possibile segnale di vendita.<br>
                                - Utilizzato per identificare potenziali inversioni di tendenza o confermare trend in corso.<br>
                            <b>Indice di Sharpe:  {sharpe_ratio} </b> <br>
                                 Misura il rapporto rischio/rendimento rispetto a un tasso privo di rischio. Un valore più alto indica una migliore performance aggiustata per il rischio.<br>
                                 - Interpretazione:  <br>
                                  * > 1: Buona performance.<br>
                                  * 0.5 - 1: Accettabile.<br>
                                  * < 0.5: Scarsa performance.<br>
                                <b>Treynor Ratio:  {treynor}</b><br>
                                Calcolato come (Rendimento del portafoglio - Rischio-free) / Beta. Misura la remunerazione del rischio sistematico (Beta). Valori più alti indicano una migliore gestione del rischio sistematico.<br>
                                <b>Sortino Ratio:  {sortino}</b><br>
                                Simile allo Sharpe, ma considera solo la volatilità negativa (downside risk). Valori più alti indicano una migliore remunerazione del rischio di perdita.<br>
                                - Interpretazione:<br>
                                  * > 2: Eccellente.<br>
                                  * 1 - 2: Accettabile.<br>
                                  * < 1: Da valutare con attenzione.<br>
                                <b>Calmar Ratio:  {calmar}</b><br>
                                Calcolato come (Rendimento annuo medio) / (Max Drawdown in valore assoluto). Misura quanto rendimento annuo si ottiene per ogni punto percentuale di drawdown. Valori più alti indicano un migliore rapporto tra guadagno e rischio di perdita.<br>

                            </p>
                                """,
                        "display": "none"}
                ]
            },
            {
                'id': 'ARIMAGARCH',
                'type': 'page_menu',
                'title': 'Modelli ARIMA e GARCH',
                'contents': [
                    {
                        "type": "plot_txt",
                        "id": "TestDF",
                        "label": "Test Dickey-Fuller",
                        "title": f"<h2> Test Dickey-Fuller </h2>",
                        "plot": plot_stationarity_html,
                        "text": f"""<p>
                                 Test Dickey-Fuller<b> (p-value):   {adf_pvalue}</b><br>
                                 Verifica la stazionarietà della serie temporale.<br>
                                 - Interpretazione del p-value:<br>
                                 * < 0.05: La serie è stazionaria (ha una media e una varianza costanti nel tempo).<br>
                                 * >= 0.05: La serie non è stazionaria.<br>
                                 Serie stazionarie sono una premessa importante per molti modelli di previsione, inclusi ARIMA e GARCH.<br>
                                 Se la serie non è stazionaria, potrebbe essere necessario differenziare o trasformare i dati.
                                 </p>
                                 """,
                    },
                    {
                        "type": "plot_table_download",
                        "id": "arima_summary",
                        "label": "Sommario Modello ARIMA",
                        "title": f"<h2>Sommario Modello ARIMA</h2>",
                        "plot": plot_arima_html,
                        "data": f"{arima_summary_html}",
                        'display': 'none'
                    },
                    {
                        "type": "plot_table_download",
                        "id": "garch_summary",
                        "label": "Sommario Modello GARCH",
                        "title": f"<h2>Sommario Modello GARCH</h2>",
                        "plot": plot_garch_html,
                        "data": f' {garch_summary_html} ',
                        'display': 'none',
                    },
                ]
            },
            {
                'id': 'AI',
                'type': 'page',
                'title': 'Modelli  AI ',
                'contents': [
                    {
                        "type": "plot_table",
                        "id": "AI",
                        "title": f"<h2>  </h2>",
                        "plot": plot_all_models_forecast_html,
                        "data": forecast_html, },
                ]

            },

            {
                'id': 'info',
                'type': 'page',
                'title': 'Info ',
                'contents': [
                    {
                        "type": "img_txt",
                        'image_src': "https://raw.githubusercontent.com/fr-cm/interfaccia/refs/heads/main/Tutorial/img/logo.webp",
                        "id": "info",
                        "text": """<p>Le analisi forniscono una panoramica completa delle performance e dei rischi di un titolo, combinando metodi statistici, machine learning e grafici avanzati. Partono dal confronto tra prezzo medio storico e prezzo recente, 
                        utile per identificare tendenze. Si calcolano il momentum, per valutare la forza dei trend, e le medie mobili esponenziali, per individuare tendenze a breve e lungo termine. La volatilità annualizzata e il massimo drawdown misurano 
                        il rischio e la stabilità del titolo, mentre il VaR e il CVaR quantificano le perdite potenziali.<br>
                        Gli indici Sharpe, Treynor, Sortino e Calmar valutano il rendimento rispetto al rischio. Beta e alpha analizzano la sensibilità e la sovra/sottoperformance rispetto al benchmark, mentre correlazione e cointegrazione esplorano legami 
                        di lungo termine. Modelli ARIMA e GARCH prevedono prezzi futuri e stimano volatilità, supportati da previsioni con intelligenza artificiale. Grafici interattivi e statici completano l'analisi, offrendo strumenti chiave per decisioni 
                        di investimento informate.<br>
                        <b>ATTENZIONE!<br> <i>Questa analisi è generata automaticamente dallo script di F.Cam. L'autore declina ogni responsabilità per l'accuratezza, la precisione, l'uso delle informazioni e eventuali malfunzionamenti dello script</i>           
                        </p>"""
                    }
                ]
            },
            {
                'id': 'footer',
                'type': 'footer',
                'contents': [
                    {'type': 'footer',  # if you want to insert html directly put html
                     'text': 'Questa analisi è stata generata automaticamente dallo script di F.Cam. L\'autore declina espressamente ogni responsabilità per eventuali errori, omissioni o inesattezze presenti nelle analisi e nelle informazioni fornite. L\'utilizzo di tali dati è a esclusivo rischio dell\'utente. L\'autore non garantisce l\'accuratezza, la completezza o l\'affidabilità delle informazioni presentate né l\'assenza di malfunzionamenti dello script utilizzato. Inoltre, l\'autore non sarà ritenuto responsabile per eventuali danni diretti, indiretti, incidentali o consequenziali derivanti dall\'uso o dall\'affidamento su queste analisi. Si consiglia di verificare e convalidare autonomamente le informazioni prima di procedere con decisioni basate su di esse.'}
                    # text footer
                ]
            }
        ]
    }

    # funzione per far stampare la dashboard
    more_dash = input(
        f"{BOLD}{'\nVuoi una dashboard con i risultatie grafici più precisi e interattivi? (SI/NO):  '}{END}").strip().lower()
    if more_dash in ['S', 'SI', 'YES', 'Y', 's', 'si', 'yes', 'y']:
        generate_dashboard(ticker, data_ds)
        print("")

    # disclamer responsabilità
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(f"{BOLD}{'                                                          ATTENZIONE '}{END}")
    slow_print("""
Questa analisi è stata generata automaticamente dallo script di F.Cam. L'autore declina espressamente ogni responsabilità \n
per eventuali errori, omissioni o inesattezze presenti nelle analisi e nelle informazioni fornite. L'utilizzo di tali dati \n
è a esclusivo rischio dell'utente. L'autore non garantisce l'accuratezza, la completezza o l'affidabilità delle informazioni \n
presentate né l'assenza di malfunzionamenti dello script utilizzato. Inoltre, l'autore non sarà ritenuto responsabile per \n
eventuali danni diretti, indiretti, incidentali o consequenziali derivanti dall'uso o dall'affidamento su queste analisi. \n
Si consiglia di verificare e convalidare autonomamente le informazioni prima di procedere con decisioni basate su di esse.""")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'\n ******************************************************   FINE ANALISI  ****************************************************** \n'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(" ")
    print(" ")


#  Loop di esecuzione script

def main():
    while True:
        run_analyses()
        messaggio = "Elaborazione..."
        stop_event = threading.Event()
        BOLD = "\033[1m"
        END = "\033[0m"
        ripeti = input(f"{BOLD}{"Vuoi eseguire un'altra analisi? (SI/NO):  "}{END}").strip().upper()
        if ripeti not in ['S', 'SI', 'YES', 'Y', 's', 'si', 'yes', 'y']:
            print(" ")
            print(" ")
            print(f"{BOLD}{"Arrivederci e tante buone cose"}{END}")
            print(" ")
            print(" ")
            break


if __name__ == "__main__":
    main()
