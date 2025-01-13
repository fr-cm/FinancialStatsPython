from analysis.DataRecall import get_benchmark_data, spinner, get_default_dates, get_stock_info, get_logo_url, slow_print, \
    get_stock_data, \
    generate_dashboard, get_valid_date
from analysis.PriceAnalysis import calculate_daily_returns, calculate_moving_averages, calculate_average_price, \
    calculate_ema, \
    calculate_momentum, calculate_average_volume
from analysis.RiskAnalysis import calculate_volatility, calculate_rsi, calculate_bollinger_bands, calculate_max_drawdown, \
    calculate_sharpe_ratio, calculate_var, calculate_sortino_ratio, calculate_treynor_ratio, calculate_calmar_ratio, \
    calculate_historical_var, calculate_beta_alpha, calculate_expected_shortfall
from analysis.BetaBenchmarkAnalysis import calculate_beta, calculate_correlation_and_cointegration
from analysis.PredictionAnalysis import make_stationary, select_arima_model, apply_garch, check_stationarity
from EN_code.charts.PlotlyChart import plot_var_analysis, plot_residual, plot_seasonality, plot_trend, \
    plot_stationarity, plot_arima_predictions, plot_garch_volatility, plot_momentum, plot_average_volume, \
    plot_moving_averages, plot_rsi_plotly, plot_bollinger_bands_plotly, plot_daily_returns_plotly, \
    perform_seasonal_decomposition_plotly, plot_data_plotly, plot_story_plotly, plot_histogram_plotly, \
    plot_correlation_plotly, plot_all_models_forecast, plot_adfuller_test, plot_seasonality_and_residual
from EN_code.charts.TerminalChart import plot_adfuller_test, plot_rsi_plotext, plot_bollinger_bands_plotext, \
    perform_seasonal_decomposition_plotext
from EN_code.analysis.PredictionAI_1 import run_ai_price_forecast, calculate_split_idx, build_forecast_table
import pandas as pd
import plotext as plt
import plotly.io as pio
from datetime import datetime
import logging

# Avoid error messages when importing TensorFlow

import os

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = show all logs, 1 = suppress WARNING, 2 = suppress INFO, 3 = suppress everything except ERROR

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
import os

# Try disabling oneDNN plugins
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Hide logs of level INFO, WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging

tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
import sys
import threading

# Printing styles for the terminal
BOLD = "\033[1m"
END = "\033[0m"


# check download benchmark
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
        print(f"\n{BOLD}ERROR:{END} During the download/analysis of the benchmark '{current_benchmark}': {e}")
        return None


#
def run_analyses():
    """
    Main function for analyzing a stock and a benchmark with dashboard output.
    """
    global benchmark_data
    BOLD = "\033[1m"
    END = "\033[0m"
    print(f"""{BOLD}{"""******************************************************************************************************************************
***************************************            FINANCIAL STATS BY F.CAM            ***************************************
******************************************************************************************************************************
"""}{END}""")
    slow_print(f"""{BOLD}{"""                               |                                      +---------------------------+
                               |           /\\                         |  DATA  |  VALUE   |  VAR  |
                               |          /  \\      /\\                |--------|----------|-------|
                               |         /    \\    /  \\/              |   ***  |    **    |   *   |
                               |        /      \\  /                   |   *    |    ***   |  **   |
                               |       /        \\/                    |   **   |    *     |  **   |
                               |      /                               |        |          |       |
                               +--------------------------------+     +---------------------------+"""}{END}""")
    print(f"""{BOLD}{"""******************************************************************************************************************************
******************************************************************************************************************************
            A TICKER IS A SERIES OF LETTERS USED TO UNIQUELY IDENTIFY THE STOCKS OF A COMPANY OR A MARKET INDEX. 
 FOR YAHOO FINANCE, YOU MUST ENTER THE EXACT TICKER FOUND ON THE WEBSITE, VALID FOR BOTH COMPANIES AND MARKET BENCHMARKS.
                                       (once the data is entered, press enter)
                                               Search for the TICKER:
                                             example: XYZ Ltd (TICKER)
                                        https://finance.yahoo.com/lookup/?s=
******************************************************************************************************************************
"""}{END}""")

    # imput defoult value
    default_ticker = ""
    default_benchmark = "ACWI"
    default_start_date = None
    default_end_date = None

    # ticker Input stock
    ticker = input(f"{BOLD} >>> {END} Enter the stock ticker  (e.g. AAPL): ").strip()

    # Calculate default dates based on the historical data of the given stock input
    default_start_date, default_end_date = get_default_dates(ticker)
    # Default dates if not present in the stock data
    if default_start_date is None or default_end_date is None:
        default_start_date = "2020-01-01"
        default_end_date = datetime.now().strftime('%Y-%m-%d')

    # benchmark ticker Input
    benchmark_input = input(
        f"{BOLD} >>> {END} Enter the benchmark ticker (optional, e.g., ^GSPC, leave blank to use the default '{default_benchmark}'):  ").strip()

    # Use default_benchmark
    benchmark = benchmark_input if benchmark_input else default_benchmark

    # Date validation function
    start_date = get_valid_date(
        f"{BOLD} >>> {END} Enter the start date (YYYY-MM-DD) [Default: {default_start_date}]:  ",
        default_start_date)
    end_date = get_valid_date(
        f"{BOLD} >>> {END} Enter the end date (YYYY-MM-DD) [Default: {default_end_date}]:  ",
        default_end_date)

    # Fetches the stock logo for the dashboard
    logo_url = get_logo_url(ticker)
    # download data stock
    data = get_stock_data(ticker, start_date, end_date)

    # Validation of the `data` result
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError(f"Error: 'get_stock_data' did not return a valid DataFrame for the ticker {ticker}.")
    if data.empty:
        raise ValueError(f"Error: No data available for the ticker {ticker}.")
    if 'close' not in data.columns:
        raise ValueError(f"Error: The DataFrame does not contain the column 'close'.")

    # Initialize benchmark_data to avoid errors
    benchmark_data = None

    # Function to attempt downloading the benchmark
    if benchmark:
        benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)
        if benchmark_data is None:
            while True:
                benchmark_input = input(
                    f"{BOLD} >>> {END} Enter a valid ticker for the benchmark (optional, leave blank to use the default '{default_benchmark}'):  "
                ).strip()

                # If the user leaves it blank, use the default_benchmark
                if not benchmark_input:
                    print(
                        f"{BOLD}INFO:{END} No input provided. Using the default benchmark. '{default_benchmark}'.")
                    benchmark = default_benchmark
                else:
                    benchmark = benchmark_input

                # Attempt to download the benchmark with the new input
                benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)

                if benchmark_data is not None:
                    break
                else:
                    # If left blank and attempted to use the default_benchmark, try downloading it
                    if not benchmark_input:
                        print(
                            f"{BOLD}INFO:{END} No input provided. Using the default benchmark. '{default_benchmark}'.")
                        benchmark = default_benchmark
                        benchmark_data = download_and_validate_benchmark(benchmark, start_date, end_date)
                        if benchmark_data is not None:
                            break

    #############################################################################
    #     ANALYSIS
    #############################################################################

    average_price = calculate_average_price(data)
    last_price = data['close'].iloc[-1] if not data['close'].empty else None
    if last_price is not None:
        price_position = "above" if last_price > average_price else "under"
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
    adf_pvalue = adf_result[1]  # p-value of test adf

    if adf_pvalue >= 0.05:
        slow_print("The series is non-stationary.")
        slow_print("Applying differentiation.")
        data = make_stationary(data)
        stop_event = threading.Event()

        #############################################################################
        #     initialization animation loading
        #############################################################################
        messaggio = (f"{BOLD} {' >>> '}{END} Processing ")
        spinner_thread = threading.Thread(target=spinner, args=(messaggio, stop_event,))
        spinner_thread.start()


    else:
        slow_print("The series is stationary.")

    # stationary Test

    # Calculation of beta with benchmark
    beta = None
    correlation = None
    coint_pvalue = None
    if benchmark_data is not None:
        beta = calculate_beta(data, benchmark_data)
        correlation, coint_pvalue = calculate_correlation_and_cointegration(data, benchmark_data)

    #  Apply ARIMA with custom function (auto search of p, d, q) instead of :
    arima_model, best_order = select_arima_model(data['close'])

    if arima_model is not None:
        arima_summary_html = ''.join([table.as_html() for table in arima_model.summary().tables])
        print(f"Best ARIMA model: ARIMA{best_order}")
    else:
        arima_summary_html = "ARIMA not available."
        print("No valid ARIMA model found.")

    # GARCH
    garch_result = apply_garch(data)
    # CVaR
    cvar_95 = calculate_expected_shortfall(data, confidence_level=0.95)
    # Sortino
    sortino = calculate_sortino_ratio(data, risk_free_rate=0.01, target_return=0.0)
    # Calmar
    calmar = calculate_calmar_ratio(data)
    # Beta & Alpha + Treynor Ratio
    # Assumes to have an array/Series benchmark_returns with the daily returns of an index
    if benchmark_data is not None:
        beta, alpha = calculate_beta_alpha(data, benchmark_data['daily_returns'], risk_free_rate=0.01)
        treynor = calculate_treynor_ratio(data, benchmark_data['daily_returns'], risk_free_rate=0.01)
    else:
        beta, alpha, treynor = None, None, None

    # Historical VaR
    var_h = calculate_historical_var(data, confidence_level=0.95)

    # Plotly graph generation
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

    # Conversion of charts to HTML
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

    # Takes general stock info and arranges it in table
    general_info = get_stock_info(ticker)
    if not general_info:
        print("Errore: Unable to retrieve general title information.")
        return
    analysis_results = {
        'general_info': general_info,
    }
    general_info_no_desc = general_info.copy()
    general_info_no_desc.pop('description', None)
    df = pd.DataFrame([general_info_no_desc])
    general_info_no_desc['Average Price'] = average_price
    general_info_no_desc['Last Price'] = last_price
    df = pd.DataFrame.from_dict(general_info_no_desc, orient='index', )
    general_info_table = df.to_html(header=False, )

    # Preparation of data for return
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

        # HTML charts
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
        data = calculate_average_volume(data, period=50)  # data stay DataFrame
        analysis_results['average_volume'] = data['avg_volume_50'].iloc[-1]

        # end loading animation
        stop_event.set()
        spinner_thread.join()
        sys.stdout.write('\r' + ' ' * (len(messaggio) + 2) + '\r')
        sys.stdout.flush()

    ##################################################################
    # OUTPUT
    ##################################################################

    print("")
    print("")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{f'                                                      ANALYSIS RESULTS FOR {ticker}\n                                                     ({start_date} -> {end_date})   '}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print("")
    print(f"Name: {general_info['longName']}")
    print(f"Sector: {general_info['sector']}")
    print(f"Industry: {general_info['industry']}")
    print(f"MarketCap: {general_info['marketCap']}")
    print(f"DividendYield: {general_info['dividendYield']}")
    print(f"Beta: {general_info['beta']}")
    print(f"FullTimeEmployees: {general_info['fullTimeEmployees']}")
    print(f"Country: {general_info['country']}")
    print(f"WebSite: {general_info['website']}")
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

    # 1) Average Price
    print(f"1) Average Price: {BOLD}{average_price:.2f}{END} €")
    print(" ")
    slow_print("   - Represents the average closing price over the entire analysis period.")
    slow_print(
        "   - A higher average price indicates an upward trend, while a lower one might suggest a downward trend.")
    slow_print("   - **Reference range:**")
    slow_print("     * > €150: High, possible overvaluation")
    slow_print("     * €100 - €150: Moderate, stable")
    slow_print("     * < €100: Low, potential buying opportunity")
    slow_print(
        "   - Useful for comparing the current price to the historical average and assessing relative performance.\n")
    slow_print(" ")

    # 2) Last Price
    if last_price is not None:
        print(f"2) Last Price: {BOLD}{last_price:.2f}{END} € ({price_position} compared to the average price)")
        print(" ")
        slow_print("   - Compares the most recent closing price available with the average closing price.")
        slow_print("   - If the last price is higher than the average, it could indicate a recent positive trend.")
        slow_print("   - If lower, it might suggest downward pressure or a market correction.")
        slow_print("   - **Interpretation of the position relative to the average:**")
        slow_print("     * > 0: Above the average, positive signal")
        slow_print("     * < 0: Below the average, negative signal")
        slow_print(" ")
    else:
        print("2) Last Price not available.")
        print("   - This could be due to missing data or an error in retrieving the information.\n")
        print(" ")

    # 3) Momentum
    print(f"\n3) Momentum (14 days): {BOLD}{analysis_results['momentum']:.2f}{END} €")
    print(" ")
    slow_print("   - Measures the speed of price change over a specific period.")
    slow_print("   - Positive values indicate an upward trend.")
    slow_print("   - Negative values indicate a downward trend.")
    slow_print("   - **Reference range:**")
    slow_print("     * > 0: Positive momentum, possible continuation of the trend.")
    slow_print("     * < 0: Negative momentum, possible trend reversal.")
    slow_print("   - Useful for confirming or anticipating price movements.\n")
    slow_print(" ")

    # 4) Exponential Moving Averages (EMA)
    print(f"\n4) EMA (Exponential Moving Averages):")
    print(f"   EMA 50: {BOLD}{analysis_results['ema_50']:.2f}{END} €")
    print(f"   EMA 200: {BOLD}{analysis_results['ema_200']:.2f}{END} €")
    print(" ")
    slow_print("   - The EMA gives more weight to recent data compared to the simple moving average (SMA).")
    slow_print("   - Used to identify short- and long-term trends.")
    slow_print("   - **Reference range:**")
    slow_print("     * EMA 50 above EMA 200: Uptrend (Golden Cross).")
    slow_print("     * EMA 50 below EMA 200: Downtrend (Death Cross).")
    slow_print("   - An EMA closer to the current price suggests higher responsiveness to price changes.")
    slow_print(" ")

    # 5) Average Volume
    if 'average_volume' in analysis_results and analysis_results['average_volume']:
        print(f"\n5) Average Volume (50 days): {BOLD}{analysis_results['average_volume']:.0f}{END}")
        print(" ")
        slow_print("   - Volume represents the number of trades executed over a specific period.")
        slow_print("   - High average volume indicates greater investor interest.")
        slow_print("   - **Interpretation:**")
        slow_print("     * Volume increases accompanied by price movements suggest trend strength.")
        slow_print("     * Volume decreases indicate weakening of the trend or lack of interest.")
        slow_print("   - Useful for assessing the strength of a price movement and investor participation.\n")
        slow_print(" ")

    # 6) Annualized Volatility

    print(f"\n6) Annualized Volatility: {BOLD}{volatility:.2f}{END}%")
    print(" ")
    slow_print("   - Measures the dispersion of annualized daily returns.")
    slow_print("   - **Reference range:**")
    slow_print("     * > 30%: High volatility, high risk")
    slow_print("     * 15% - 30%: Moderate volatility, balanced risk")
    slow_print("     * < 15%: Low volatility, low risk")
    slow_print(
        "   - High volatility indicates greater risk and potential for significant price swings.")
    slow_print("   - Low volatility suggests a more stable market.\n")
    slow_print(" ")

    # 7) Maximum Drawdown
    print(f"\n7) Maximum Drawdown: {BOLD}{max_drawdown:.2%}{END}")
    print(" ")
    slow_print("   - Represents the maximum percentage loss from the highest peak to the lowest trough in the period.")
    slow_print("   - **Reference range:**")
    slow_print("     * > 50%: High risk of significant loss")
    slow_print("     * 20% - 50%: Moderate risk")
    slow_print("     * < 20%: Low risk of loss")
    slow_print("   - Indicates the maximum potential loss risk during the investment.")
    slow_print("   - Useful for assessing the resilience of the stock during adverse market periods.\n")
    print(" ")

    # 8) Value at Risk (VaR)
    print(f"\n8) Value at Risk (95%): {BOLD}{-var_value:.2%}{END}")
    print(" ")
    slow_print("    - Estimates the maximum expected loss (as a percentage) with a 95% confidence level.")
    slow_print("    - **Interpretation of VaR:**")
    slow_print(
        "      * A 5% VaR indicates there is only a 5% probability that the loss will exceed this value in a given period.")
    slow_print(
        "    - Useful for evaluating the potential risk of an investment and implementing risk management strategies.\n")
    print(" ")

    # 9) Historical VaR (95%)
    print(f"\n9) Historical VaR (95%): {BOLD}{var_h:.4f}{END}")
    print(" ")
    slow_print("   - VaR indicates the worst expected loss (or loss threshold) at a certain confidence level.")
    slow_print("   - This 'historical' version does not assume any distribution (e.g., normal).")
    slow_print("   - It is simply based on the worst past performances.\n")

    # 10) Expected Shortfall / CVaR (95%)
    print(f"\n10) Expected Shortfall (CVaR) 95%: {BOLD}{cvar_95:.4f}{END}")
    print(" ")
    slow_print("   - CVaR indicates the average (expected) loss in case the VaR is exceeded.")
    slow_print(
        "   - Compares to VaR: while VaR tells you the 'threshold,' CVaR tells you 'how much is lost on average' beyond that threshold.")
    slow_print("   - **Negative values** → If interpreted as returns, they correspond to losses.")
    slow_print("   - The larger the CVaR (in absolute negative value), the heavier the loss tail.\n")

    # 11) Sharpe Ratio
    print(f"\n11) Sharpe Ratio: {BOLD}{sharpe_ratio:.2f}{END}")
    print(" ")
    slow_print("   - Risk/reward ratio relative to a risk-free rate.")
    slow_print("   - **Interpretation of Sharpe Ratio:**")
    slow_print("     * > 1: Good risk-adjusted performance")
    slow_print("     * 0.5 - 1: Acceptable performance")
    slow_print("     * < 0.5: Poor risk-adjusted performance")
    slow_print("   - A higher Sharpe ratio indicates better risk-adjusted performance.")
    slow_print(
        "   - Helps compare the efficiency of different investments in terms of return per unit of risk.\n")
    print(" ")

    # 12) Treynor Ratio
    print(f"\n12) Treynor Ratio: {BOLD}{treynor:.2f}{END}")
    slow_print("   - Treynor = (Portfolio return - Risk-free rate) / Beta")
    slow_print("   - Unlike the Sharpe Ratio, it only considers systematic risk (Beta).")
    slow_print("   - The higher the Treynor, the better the compensation for systematic risk.\n")

    # 13) Sortino Ratio
    print(f"\n13) Sortino Ratio: {BOLD}{sortino:.2f}{END}")
    slow_print(
        "   - The Sortino Ratio is similar to the Sharpe Ratio but only considers negative volatility (downside risk).")
    slow_print("   - The higher it is, the better the portfolio compensates for downside risk.")
    slow_print("   - **Indicative range:**")
    slow_print("     * Sortino > 2: Excellent")
    slow_print("     * 1 < Sortino < 2: Acceptable")
    slow_print("     * < 1: Requires closer evaluation\n")

    # 14) Calmar Ratio
    print(f"\n14) Calmar Ratio: {BOLD}{calmar:.2f}{END}")
    slow_print("   - Calmar = (Annual average return) / (Absolute value of maximum drawdown).")
    slow_print("   - Indicates how much 'annual return' is earned for each percentage point of drawdown.")
    slow_print("   - The higher it is, the better the ratio between gain and risk of loss.\n")

    # 15) Beta and Alpha
    print(f"\n15) Beta: {BOLD}{beta:.2f}{END}, Alpha (annual): {BOLD}{alpha:.4f}{END}")
    slow_print("   - **Beta** measures the sensitivity of your asset/portfolio to the market (benchmark).")
    slow_print("     * Beta = 1: Moves in line with the market")
    slow_print("     * Beta > 1: Amplifies market movements (more risky)")
    slow_print("     * Beta < 1: Less sensitive to market fluctuations (less risky)")
    slow_print("   - **Alpha** measures the excess (or deficit) return not explained by Beta.")
    slow_print("     * Positive Alpha = Outperformance relative to market risks")
    slow_print("     * Negative Alpha = Underperformance\n")

    # 16) Beta relative to the benchmark
    if beta is not None:
        print(f"\n16) Beta relative to the benchmark ({benchmark}): {BOLD}{beta:.2f}{END}")
        print(" ")
        slow_print("   - Indicates how the stock moves in relation to the reference benchmark.")
        slow_print("   - **Interpretation of Beta:**")
        slow_print(
            "     * > 1: The stock is more volatile than the benchmark, indicating higher risk and return potential.")
        slow_print("     * < 1: The stock is less volatile than the benchmark, suggesting lower risk.")
        slow_print("     * = 1: The stock moves in line with the benchmark.")
        print(" ")
    else:
        slow_print("\n16) Beta: Not available (benchmark not provided or invalid).")
        slow_print("   - Ensure you have provided a valid benchmark to calculate Beta.\n")
        print(" ")

    # 17) Correlation and Cointegration with the benchmark
    if correlation is not None and coint_pvalue is not None:
        print(f"\n17) Correlation with the benchmark: {BOLD}{correlation:.4f}{END}")
        print(" ")
        slow_print(
            "   - Indicates the degree of linear relationship between the stock's returns and the benchmark's returns.")
        print(" ")
        if correlation > 0:
            slow_print("   - Positive correlation: returns tend to move in the same direction.")
            print(" ")
        else:
            slow_print("   - Negative correlation: returns tend to move in opposite directions.")
        print(f"   Cointegration P-value: {BOLD}{coint_pvalue:.4f}{END}")
        print(" ")
        if coint_pvalue < 0.05:
            slow_print("   - Value < 0.05 indicates significant cointegration (long-term relationship).")
            slow_print(
                "   - This means there is a stable relationship between the stock and the benchmark over the long term.")
            print(" ")
        else:
            slow_print("   - Value >= 0.05 indicates insufficient evidence of significant cointegration.")
            print(" ")
    else:
        slow_print("\n17) Correlation and Cointegration: Not available (benchmark not provided or invalid).")
        slow_print(
            "   - Ensure you have provided valid data for the benchmark and that the calculation was successful.\n")
        print(" ")

    # 18) Dickey-Fuller Test
    print(f"\n18) Dickey-Fuller Test (p-value): {BOLD}{adf_pvalue:.4f}{END}")
    print(" ")
    slow_print("    - Tests the stationarity of the time series.")
    slow_print("    - **Interpretation of the p-value:**")
    slow_print("      * < 0.05: The series is stationary (has a constant mean and variance over time).")
    slow_print("      * >= 0.05: The series is not stationary.")
    slow_print(
        "    - Stationary series are an important prerequisite for many forecasting models, including ARIMA and GARCH.")
    slow_print("    - If the series is not stationary, it may be necessary to differentiate or transform the data.\n")
    print(" ")

    # 19) ARIMA Model Summary
    print(f"\n19) ARIMA(1,1,1) - Model Summary:\n{arima_model.summary()}")
    print(" ")
    slow_print("   - The ARIMA model combines autoregression (AR) and moving average (MA) to forecast future prices.")
    slow_print("   - **Key components in the summary:**")
    slow_print("     * **Coefficients:** Indicate the impact of AR and MA components on the model.")
    slow_print(
        "     * **p-values:** Assess the statistical significance of each coefficient (values < 0.05 are generally considered significant).")
    slow_print(
        "     * **AIC and BIC:** Information criteria for model comparison; lower values indicate better models.")
    slow_print("   - Use this model to make forecasts on future prices based on historical data.\n")
    print(" ")

    # 20) GARCH Model Summary
    print(f"\n20) GARCH(1,1) - Model Summary:\n{garch_result.summary()}")
    print(" ")
    slow_print("   - The GARCH model estimates and forecasts the volatility of returns.")
    slow_print("   - **Key components in the summary:**")
    slow_print("     * **Omega:** Constant in the volatility model.")
    slow_print("     * **Alpha:** Impact of past market shocks on current volatility.")
    slow_print("     * **Beta:** Influence of past volatility on current volatility.")
    slow_print(
        "     * **p-values:** Assess the significance of the parameters (values < 0.05 indicate significance).")
    slow_print(
        "   - A well-calibrated GARCH model helps understand and forecast future volatility, useful for risk management.\n")
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
    print(" - Cyan: Close Price")
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
    # FUNCTION TO PROMPT THE USER FOR AI TRAINING OR NOT
    ####################################################

    use_ai = input(
        f"{BOLD}{'Would you also like to run a price forecast using Artificial Intelligence? (YES/NO): '}{END}"
    ).strip().lower()

    plot_all_models_forecast_html = "<p><b>AI analysis not performed.</b></p>"
    forecast_html = "<p></p>"

    if use_ai in ['s', 'si', 'y', 'yes','Y','YES']:

        # Executes the function that trains ALL models (LSTM, CNN+LSTM, Transformer)
        # and returns a "results" dictionary containing test and forecast data for each model
        ai_results = run_ai_price_forecast(
            data=data,
            arima_model=arima_model,
            garch_result=garch_result,
            benchmark_data=benchmark_data,
            future_steps=15  # prevision day numbers
        )

        # Checks if the function returned a valid result
        if ai_results is not None and isinstance(ai_results, dict) and len(ai_results) > 0:
            # ai_results is a dictionary like:
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

            # Calculate the split_idx to align the test portion
            split_idx = calculate_split_idx(data, test_size=0.2)

            # Generate the chart with the new plot_all_models_forecast function
            # (this function takes ALL predictions as input)
            plot_all_models_forecast_fig = plot_all_models_forecast(
                data=data,
                results=ai_results,  # Pass the complete results dictionary
                split_idx=split_idx
            )

            # Convert the chart to HTML
            plot_all_models_forecast_html = pio.to_html(
                plot_all_models_forecast_fig,
                full_html=False,
                include_plotlyjs='cdn'
            )

            # Build an HTML table for each model by concatenating the forecast DataFrames
            all_forecast_tables_html = ""
            merged_df, table_html = build_forecast_table(ai_results)

            if merged_df is not None:
                all_forecast_tables_html = table_html.replace('\n', '')
            else:
                all_forecast_tables_html = table_html  # Error or absence message

            forecast_html = all_forecast_tables_html

        else:
            plot_all_models_forecast_html = "<p>AI models not available.</p>"
            forecast_html = "<p>AI analysis not performed.</p>"

    # ASSIGNMENT OF RESULTS TO THE
    analysis_results['plot_all_models_forecast_html'] = plot_all_models_forecast_html
    analysis_results['forecast_html'] = forecast_html

    ######################################################
    # PLAINING DASHBOARD
    ######################################################
    data_ds = {
        'navbar': {
            'brand_name': f'{general_info['longName']}',
            'subtitle': f'{ticker}',
            'menu_items': [

                {
                    'id': 'Analysis', 'label': 'Analysis', 'submenu': [
                    {'id': 'price_trend', 'label': 'Price'},
                    {'id': 'Risk', 'label': 'Risk and Volatility'},
                    {'id': 'info', 'label': 'Info'},
                ]
                },
                {
                    'id': 'seasonality', 'label': 'Seasonality'
                },
                {
                    'id': 'Models', 'label': 'Predictive Models', 'submenu': [
                    {'id': 'ARIMAGARCH', 'label': 'ARIMA and GARCH'},
                    {'id': 'AI', 'label': 'AI'},
                ]
                }

            ]
        },
        'pages': [
            {
                'id': 'home',
                'type': 'home',
                'title': f'Analysis of {general_info["longName"]} <br> From {start_date} to {end_date}',
                'image_src': logo_url,
                'contents': [
                    {
                        "type": "home",
                        "id": "home",
                        "content": f"<p>{general_info['description']}</p>"
                    }
                ]
            },

            {
                'id': 'price_trend',
                'type': 'page_menu',
                'title': 'Price and Trend',
                'contents': [
                    {
                        "type": "plot_table_download",
                        "id": "Price",
                        "label": "Price",
                        "plot": plot_data_plotly_html,
                        "title": f"<h3>  </h3>",
                        "data": general_info_table,
                    },

                    {
                        "type": "plot_txt_download",
                        "id": "MovingAverage",
                        "label": "Moving Average",
                        "plot": sma_ema_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                            <p>
                            <b>Simple Moving Average (SMA)</b><br>
                            The simple moving average represents the arithmetic mean of closing prices over a given period. It provides a linear view of the price trend, reducing short-term fluctuations.<br>
                            - Benefits:<br>
                              * Identification of long- and short-term trends.<br>
                              * Dynamic levels of support and resistance.<br>
                            - Interpretation:<br>
                               A price above the SMA indicates an uptrend.<br>
                               A price below the SMA indicates a downtrend.<br>
                            <br>

                            <b>Exponential Moving Average (EMA)</b><br>
                            EMA 50: <b>{analysis_results['ema_50']:.2f}€</b>  |   EMA 200: <b>{analysis_results['ema_200']:.2f}€</b><br>
                            The exponential moving average gives more weight to recent prices compared to the SMA, making it more responsive to price changes.<br>

                               Helps identify recent trends, reversals, and signals for entry and exit in financial markets.<br>
                               EMA 50 above EMA 200: Bullish signal (Golden Cross).<br>
                               EMA 50 below EMA 200: Bearish signal (Death Cross).<br>
                            - Reference range:<br>
                               EMA 50: Short-term trends.<br>
                               EMA 200: Long-term trends.<br>
                            <br>
                            </p>
                            """,
                        'display': 'none'
                    },

                    {
                        "type": "plot_txt_download",
                        "id": "momentum",
                        "label": "Momentum",
                        "plot": momentum_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                            <p>
                            <b>Momentum:  {analysis_results['momentum']:.2f}</b><br>
                            Momentum measures the speed at which the price of a stock changes over a specific period. It is calculated by comparing the current price to the price from a set number of days ago.<br>
                            - Formula:<br>
                              <i>Momentum = Current Price - Price N Days Ago</i><br>
                            Helps identify the strength of a trend and evaluate potential trend reversals.<br>
                            - Interpretation:<br>
                               <b>Positive values:</b> Indicate an upward trend, with the price strengthening.<br>
                               <b>Negative values:</b> Indicate a downward trend, with the price losing strength.<br>
                            - Reference range:<br>
                              * > 0: Positive momentum, possible continuation of the upward trend.<br>
                              * < 0: Negative momentum, possible trend reversal or weakening.<br>
                            <br>
                            </p>
                            """,
                        'display': 'none'
                    },

                    {
                        "type": "plot_txt_download",
                        "id": "volume",
                        "label": "Average Volume",
                        "plot": volume_html,
                        "title": f"<h3>  </h3>",
                        "text": f"""
                            <p>
                            <b>Average Volume:  {analysis_results['average_volume']}</b><br>
                            Average volume represents the average number of trades executed on a stock over a specific time period. It indicates the level of investor interest and activity in the market.<br>
                               Measures market interest in a stock.<br>
                               Evaluates the strength of a trend or price movement.<br>
                            - Interpretation:<br>
                              High volume: Indicates strong investor interest and often confirms the direction of the trend.<br>
                              Low volume: May indicate uncertainty or lack of interest in the stock.<br>
                            - Reference range:<br>
                              Volume increases accompanied by price movements suggest trend strength.<br>
                              Volume decreases may indicate trend weakening or lack of participation.<br>
                            <br>
                            </p>
                            """,
                        'display': 'none'
                    },

                    {
                        "type": "plot_txt",
                        "id": "benchmark_info",
                        "label": "Benchmark Information",
                        "plot": plot_correlation_plotly_html,
                        "title": f"<h2>Benchmark Analysis: {benchmark}</h2>",
                        "text": f"""
                            <p>
                            <b>Beta: {beta}</b><br>
                            Indicates how much the stock moves in relation to the reference benchmark.<br>
                            - Interpretation:<br>
                              * > 1: More volatile than the benchmark<br>
                              * < 1: Less volatile than the benchmark<br>
                              * = 1: Moves in line with the benchmark<br>
                            <b>Correlation with the Benchmark: {correlation}</b><br>
                            - Indicates the degree of linear relationship between the stock's and the benchmark's returns.<br>
                            - Interpretation:<br>
                              * > 0: Positive correlation<br>
                              * < 0: Negative correlation<br>
                            <b>Cointegration P-value: {coint_pvalue}</b><br>
                            Evaluates the presence of a long-term relationship between the stock and the benchmark.<br>
                            - Interpretation:<br>
                              * < 0.05: Significant cointegration<br>
                              * >= 0.05: No evidence of significant cointegration<br>
                            </p>
                        """,
                        'display': 'none'
                    },

                    {
                        "type": "plot",
                        "id": "Returns",
                        "label": "Return Distribution",
                        "plot": plot_histogram_plotly_html,
                        'display': 'none'}]
                    },
                    {
                        'id': 'seasonality',
                        'type': 'page',
                        'title': '',
                        'contents': [
                            {
                                "type": "html",
                                "id": "seasonality_info",
                                "label": "Seasonal Decomposition",
                                "title": " ",
                                "html": f"""<div style=\" background-color: #e4e4e4;padding-top: 0rem; text-align: center; display:flex; flex-wrap: nowrap; justify-content:center; flex-direction: column; align-items: center;\"><h2> Seasonal Decomposition {ticker} from {start_date} to {end_date}</h2>                       
                                        <p>
                                         Seasonal decomposition breaks down the time series into three components:<br>
                                         * <b>Trend:</b> The long-term direction of the series (e.g., upward or downward trend).<br>
                                         * <b>Seasonality:</b> Regular and repetitive fluctuations tied to specific periods (e.g., seasons, months).<br>
                                         * <b>Residual:</b> The random or irregular component remaining after removing trend and seasonality.<br>
                                         Useful for identifying recurring patterns and improving modeling and forecasting.
                                         </p></div>"""
                            },
                            {
                                "type": "plot_plot",
                                "id": "seasonality_plots",
                                "plot1": plot_seasonality_and_residual_fig_html,
                                "plot2": decomposition_trend_html,
                            },
                        ]
                    },

            {
                'id': 'Risk',
                'type': 'page_menu',
                'title': 'Risk and Volatility',
                'contents': [
                    {
                        "type": "plot_txt",
                        "id": "Volatility",
                        "label": "Volatility and Drawdown",
                        "plot": plot_bollinger_bands_plotly_html,
                        "title": "",
                        "text":
                            f"""
                            <p>
                            <b>Annualized Volatility: {volatility}%</b>  <br>
                            Measures the dispersion of annualized daily returns.<br>
                            - Reference range:<br>
                              * > 30%: High volatility, high risk<br>
                              * 15% - 30%: Moderate volatility, balanced risk<br>
                              * < 15%: Low volatility, low risk<br>
                            - High volatility indicates greater risk and potential for significant price swings.<br>
                            - Low volatility suggests a more stable market.<br>
                            <b>Maximum Drawdown: {(max_drawdown * 100):.2f}%</b>  <br>
                            Represents the maximum percentage loss from the highest peak to the lowest trough in the period.<br>
                            - Reference range:<br>
                              * > 50%: High risk of significant loss<br>
                              * 20% - 50%: Moderate risk<br>
                              * < 20%: Low risk of loss<br>
                            Helps assess the resilience of the stock during adverse market periods.
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
                            <b>Value at Risk (95%): {-var_value:.2%}</b><br>
                            Estimates the maximum expected loss (as a percentage) with a 95% confidence level.
                            A 5% VaR indicates that there is only a 5% chance the loss will exceed this value in a given period.<br>
                            <b>Historical VaR (95%): {var_h:.4f}</b><br>
                            The Historical Value at Risk (VaR) at 95% represents the maximum expected loss, with a 95% confidence level, based on historical data. 
                            It does not assume any statistical distribution but uses the worst historical returns.<br>
                            - Interpretation:<br>
                               A 5% historical VaR indicates that, with 95% confidence, the loss will not exceed this value in a given period.<br>
                               Higher (negative) values indicate greater risk.<br>
                            <b>Expected Shortfall (CVaR) 95%: {cvar_95:.4f}</b><br>
                            The Expected Shortfall (or CVaR) measures the average expected loss in case the VaR threshold is exceeded. It is considered a more conservative indicator compared to VaR as it accounts for the magnitude of losses in the tail.<br>
                            - Interpretation:<br>
                               CVaR represents the average loss in extreme adverse conditions.<br>
                               The more negative the CVaR, the greater the risk of extreme loss.<br>
                            </p>
                        """,
                        "display": "none",
                    },

                    {
                        "type": "plot_txt",
                        "id": "INDEX",
                        "label": "Indices",
                        "plot": plot_rsi_plotly_html,
                        "title": "",
                        "text": f"""                                
                            <p>
                            <b>RSI (Relative Strength Index)</b><br>
                            The RSI is an oscillator that measures the strength and speed of recent price movements, providing insights into overbought or oversold conditions.<br>
                            - Interpretation:<br>
                              * < 30: Oversold condition, potential buying opportunity.<br>
                              * > 70: Overbought condition, potential sell signal.<br>
                            - Used to identify potential trend reversals or confirm ongoing trends.<br>

                            <b>Sharpe Ratio: {sharpe_ratio}</b><br>
                            Measures the risk/reward ratio relative to a risk-free rate. A higher value indicates better risk-adjusted performance.<br>
                            - Interpretation:<br>
                              * > 1: Good performance.<br>
                              * 0.5 - 1: Acceptable.<br>
                              * < 0.5: Poor performance.<br>

                            <b>Treynor Ratio: {treynor}</b><br>
                            Calculated as (Portfolio return - Risk-free rate) / Beta. Measures compensation for systematic risk (Beta). Higher values indicate better management of systematic risk.<br>

                            <b>Sortino Ratio: {sortino}</b><br>
                            Similar to the Sharpe Ratio but considers only downside risk (negative volatility). Higher values indicate better compensation for downside risk.<br>
                            - Interpretation:<br>
                              * > 2: Excellent.<br>
                              * 1 - 2: Acceptable.<br>
                              * < 1: Requires closer evaluation.<br>

                            <b>Calmar Ratio: {calmar}</b><br>
                            Calculated as (Annual average return) / (Maximum absolute drawdown). Measures how much annual return is earned for each percentage point of drawdown. Higher values indicate a better balance between gain and risk of loss.<br>
                            </p>
                        """,
                        "display": "none"
                    }

                ]
            },
            {
                'id': 'ARIMAGARCH',
                'type': 'page_menu',
                'title': 'ARIMA and GARCH Models',
                'contents': [
                    {
                        "type": "plot_txt",
                        "id": "TestDF",
                        "label": "Dickey-Fuller Test",
                        "title": f"<h2> Dickey-Fuller Test </h2>",
                        "plot": plot_stationarity_html,
                        "text": f"""<p>
                            Dickey-Fuller Test<b> (p-value):   {adf_pvalue}</b><br>
                            Tests the stationarity of the time series.<br>
                            - Interpretation of the p-value:<br>
                            * < 0.05: The series is stationary (has a constant mean and variance over time).<br>
                            * >= 0.05: The series is not stationary.<br>
                            Stationary series are an important prerequisite for many forecasting models, including ARIMA and GARCH.<br>
                            If the series is not stationary, it may be necessary to differentiate or transform the data.
                            </p>
                            """,
                    },

                    {
                        "type": "plot_table_download",
                        "id": "arima_summary",
                        "label": "ARIMA Model Summary",
                        "title": f"<h2>ARIMA Model Summary</h2>",
                        "plot": plot_arima_html,
                        "data": f"{arima_summary_html}",
                        'display': 'none'
                    },
                    {
                        "type": "plot_table_download",
                        "id": "garch_summary",
                        "label": "GARCH Model Summary",
                        "title": f"<h2>GARCH Model Summary</h2>",
                        "plot": plot_garch_html,
                        "data": f"{garch_summary_html}",
                        'display': 'none',
                    },
                ]
            },
            {
                'id': 'AI',
                'type': 'page',
                'title': 'AI Models',
                'contents': [
                    {
                        "type": "plot_table",
                        "id": "AI",
                        "title": f"<h2>AI Model Forecast</h2>",
                        "plot": plot_all_models_forecast_html,
                        "data": forecast_html,
                    },
                ]
            },

            {
                'id': 'info',
                'type': 'page',
                'title': 'Info',
                'contents': [
                    {
                        "type": "img_txt",
                        'image_src': "https://raw.githubusercontent.com/fr-cm/interfaccia/refs/heads/main/Tutorial/img/logo.webp",
                        "id": "info",
                        "text": """<p>The analysis provides a comprehensive overview of the performance and risks of a stock, combining statistical methods, machine learning, and advanced charts. It begins with a comparison of historical average prices and recent prices, 
                        useful for identifying trends. Momentum is calculated to assess trend strength, while exponential moving averages help identify short- and long-term trends. Annualized volatility and maximum drawdown measure risk and stability, while VaR and CVaR quantify potential losses.<br>
                        The Sharpe, Treynor, Sortino, and Calmar indices evaluate returns relative to risk. Beta and alpha analyze sensitivity and over/underperformance relative to the benchmark, while correlation and cointegration explore long-term relationships. ARIMA and GARCH models forecast future prices and estimate volatility, supported by AI-based predictions. Interactive and static charts complete the analysis, providing key tools for informed investment decisions.<br>
                        <b>WARNING!<br> <i>This analysis is automatically generated by F.Cam's script. The author expressly disclaims any responsibility for the accuracy, precision, use of the information, and any malfunctions of the script</i>           
                        </p>"""
                    }
                ]
            },
            {
                'id': 'footer',
                'type': 'footer',
                'contents': [
                    {
                        'type': 'footer',  # if you want to insert HTML directly, use 'html'
                        'text': 'This analysis was automatically generated by F.Cam\'s script. The author expressly disclaims any responsibility for any errors, omissions, or inaccuracies in the analyses and information provided. The use of such data is at the user\'s sole risk. The author does not guarantee the accuracy, completeness, or reliability of the presented information or the absence of malfunctions in the script used. Furthermore, the author will not be held liable for any direct, indirect, incidental, or consequential damages arising from the use or reliance on these analyses. It is recommended to independently verify and validate the information before proceeding with decisions based on it.'
                    }
                ]
            }

        ]
    }

    # funzione per far stampare la dashboard
    more_dash = input(
        f"{BOLD}{'\nWould you like a dashboard with more detailed and interactive results and charts? (YES/NO): '}{END}").strip().lower()
    if more_dash in ['S', 'SI', 'YES', 'Y', 's', 'si', 'yes', 'y']:
        generate_dashboard(ticker, data_ds)
        print("")

    # disclamer
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(f"{BOLD}{'                                                          WARNING '}{END}")
    slow_print("""
    This analysis was automatically generated by F.Cam's script. The author expressly disclaims any responsibility 
    for errors, omissions, or inaccuracies present in the analyses and the information provided. The use of such data 
    is at the user's sole risk. The author does not guarantee the accuracy, completeness, or reliability of the presented 
    information or the absence of malfunctions in the script used. Furthermore, the author will not be held liable for 
    any direct, indirect, incidental, or consequential damages arising from the use or reliance on these analyses. 
    It is recommended to independently verify and validate the information before making decisions based on it.
    """)

    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(
        f"{BOLD}{'\n ******************************************************   END ANALYSIS  ****************************************************** \n'}{END}")
    print(
        f"{BOLD}{'******************************************************************************************************************************'}{END}")
    print(" ")
    print(" ")


# Script Execution Loop
def main():
    while True:
        run_analyses()
        message = "Processing..."
        stop_event = threading.Event()
        BOLD = "\033[1m"
        END = "\033[0m"
        repeat = input(f"{BOLD}{'Would you like to perform another analysis? (YES/NO):  '}{END}").strip().upper()
        if repeat not in ['S', 'SI', 'YES', 'Y', 's', 'si', 'yes', 'y']:
            print(" ")
            print(" ")
            print(f"{BOLD}{'Goodbye and best wishes'}{END}")
            print(" ")
            print(" ")
            break



if __name__ == "__main__":
    main()
