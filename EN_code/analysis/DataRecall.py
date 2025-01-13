import sys
import time
import yfinance as yf
from urllib.parse import urlparse
from deep_translator import GoogleTranslator
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import webbrowser
import requests
import logging
from importlib.metadata import version, PackageNotFoundError
import itertools

#Traslator set
translator = GoogleTranslator(source='auto', target='it')

########################################################################################################
########################################################################################################
#                                            get data                                                  #
########################################################################################################
########################################################################################################


def get_package_version(package_name):
    try:
        pkg_version = version(package_name)
        return pkg_version
    except Exception:
        return None
from datetime import datetime


# Update default dates based on available data
def get_default_dates(ticker):
    """
    Retrieves the first and last available dates for a ticker from Yahoo Finance.
    """
    try:
        # Download all available data
        data = yf.download(ticker, period="max")
        if not data.empty:
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            return start_date, end_date
        else:
            raise ValueError("Historical data not available for the ticker.")
    except Exception as e:
        print(f"Error retrieving default dates: {e}")
        return None, None


def get_valid_date(prompt, default):
    """
    Prompts the user to input a date in the format YYYY-MM-DD.
    If the input is empty, returns the default date.
    If the input is invalid, prompts the user again.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            print(f"     : No date entered. Using the default date: {default}")
            return default
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d'):
            try:
                parsed_date = datetime.strptime(user_input, fmt)
                formatted_date = parsed_date.strftime('%Y-%m-%d')
                print(f"     : Valid date entered: {formatted_date}")
                return formatted_date
            except ValueError:
                continue
        # If no format works, inform the user and retry
        print("Error: Please enter a valid date in the format YYYY-MM-DD (e.g., 2023-01-31).")



########################################################################################################
########################################################################################################
#                                            SLOW_PRINT                                                #
########################################################################################################
########################################################################################################
def slow_print(text, delay=0.002):
    """
    Prints text character by character with a delay.
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')  # Adds a new line at the end


def slow_print_line(text, delay=0.002):
    """
    Prints text line by line with a delay between lines.
    """
    print(text)
    time.sleep(delay)


########################################################################################################
########################################################################################################
#                                            LOADING ANIMATION                                         #
########################################################################################################
########################################################################################################
def spinner(messaggio, stop_event):
    """
    loading animation
    """
    spinner_sequence = ['-   ', '\\   ', '|   ', '/   ','-   ']
    ciclo_spinner = itertools.cycle(spinner_sequence)
    while not stop_event.is_set():
        current_spinner = next(ciclo_spinner)
        sys.stdout.write(f'\r{messaggio} {current_spinner}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(messaggio) + 2) + '\r')
    sys.stdout.flush()



########################################################################################################
########################################################################################################
#                                            GENERATE DASHBOARD                                        #
########################################################################################################
########################################################################################################
def generate_dashboard(ticker, data_ds):
    """
    Generates an HTML dashboard using the template from GitHub,
    saves it in the Downloads folder, and opens it in the browser.
    """
    try:
        # Template URL
        template_url = "https://raw.githubusercontent.com/fr-cm/interfaccia/main/Code/Template_interfaccia.html"

        # Download the template
        response = requests.get(template_url)
        if response.status_code != 200:
            logging.error(f"Unable to download the template. HTTP Status: {response.status_code}")
            print(f"Error: Unable to download the template from the provided link. HTTP Status: {response.status_code}")
            return

        template_content = response.text

        # Create a Template object
        template = Template(template_content)

        # Render the template with the data, passing them as 'data'
        rendered_html = template.render(data=data_ds)

        # Locate the Downloads folder cross-platform
        home = Path.home()
        download_dir = home / "Downloads"

        if not download_dir.exists():
            logging.warning(f"The Downloads folder does not exist: {download_dir}. The home directory will be used.")
            download_dir = home

        # Output file path
        output_filename = f"{ticker}_dashboard.html"
        output_path = download_dir / output_filename

        # Write the rendered HTML to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        logging.info(f"Dashboard successfully generated: {output_path}")
        print(f"Dashboard successfully generated: {output_path}")

        # Automatically open the dashboard in the browser
        try:
            file_url = output_path.resolve().as_uri()
            webbrowser.open(file_url)
            logging.info(f"Dashboard automatically opened in the browser: {file_url}")
            print(f"Dashboard automatically opened in the browser: {file_url}")
        except Exception as e:
            logging.error(f"Error opening the dashboard in the browser: {e}")
            print(f"Error opening the dashboard in the browser: {e}")

    except Exception as e:
        logging.error(f"Error during dashboard generation: {e}")
        print(f"Error during dashboard generation: {e}")


########################################################################################################
########################################################################################################
#                                            GET_STOCK_INFO                                            #
########################################################################################################
########################################################################################################
def get_stock_info(ticker):
    """
    Retrieves general stock information such as market capitalization, sector, industry, etc.
    If the description is unavailable, uses a default text.
    """
    default_description = (
        "The analysis provides a comprehensive overview of a stock's performance and risks, combining statistical methods, "
        "machine learning, and advanced charts. It starts with a comparison of historical average prices and recent prices, "
        "useful for identifying trends. Momentum is calculated to evaluate trend strength, and exponential moving averages "
        "are used to identify short- and long-term trends. Annualized volatility and maximum drawdown measure risk and stability, "
        "while VaR and CVaR quantify potential losses. The Sharpe, Treynor, Sortino, and Calmar indices assess return relative "
        "to risk. Beta and alpha analyze sensitivity and over/underperformance relative to the benchmark, while correlation and "
        "cointegration explore long-term relationships. ARIMA and GARCH models forecast future prices and estimate volatility, "
        "supported by AI-driven predictions. Interactive and static charts complete the analysis, providing key tools for informed "
        "investment decisions."
    )

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # Recupera la descrizione originale
        description = info.get('longBusinessSummary', '').strip()
        logging.debug(f"Descrizione originale per {ticker}: {description}")

        if description and description.lower() not in ['n/a', 'not available', '']:
            description_finale = description
        else:
            logging.warning(f"Nessuna descrizione disponibile per il ticker {ticker}. Utilizzo del testo predefinito.")
            description_finale = default_description

        # Recupera il nome completo, con fallback al ticker se non disponibile
        long_name = info.get('longName', '').strip()
        if not long_name or long_name.lower() in ['n/a', 'not available', '', 'n/a']:
            long_name = ticker  # Fallback al ticker

        general_info = {
            'longName': long_name,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'fullTimeEmployees': info.get('fullTimeEmployees', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'description': description_finale,
        }
        return general_info
    except Exception as e:
        logging.error(f"Error retrieving general information for ticker {ticker}: {e}")
        return {}



########################################################################################################
########################################################################################################
#                                            GET_LOGO_URL                                              #
########################################################################################################
########################################################################################################
def get_logo_url(ticker):
    """
    Retrieves the URL of the company's logo associated with the provided ticker.
    Sets a default logo if the company logo is unavailable.
    """
    default_logo_url = "https://raw.githubusercontent.com/fr-cm/interfaccia/refs/heads/main/Tutorial/img/interfaccia_1.png"
    try:
        ticker_info = yf.Ticker(ticker).info
        website = ticker_info.get('website')
        if website:
            # Extract the domain from the website URL
            parsed_url = urlparse(website)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            logo_url = f"https://logo.clearbit.com/{domain}"

            # Check if the logo exists
            response = requests.get(logo_url)
            if response.status_code == 200:
                return logo_url
            else:
                logging.warning(f"Logo not found for domain {domain}. Using default logo.")
                return default_logo_url
        else:
            logging.warning(f"Website not found for ticker {ticker}. Using default logo.")
            return default_logo_url
    except Exception as e:
        logging.error(f"Error retrieving logo for ticker {ticker}: {e}")
        return default_logo_url




########################################################################################################
########################################################################################################
#                                            GET_STOCK_DATA                                            #
########################################################################################################
########################################################################################################
def get_data(ticker, start, end, tipo='Titolo'):
    """
    Fetches data for a given ticker within a specified date range. Handles errors and prompts for valid tickers if necessary.

    Parameters:
        ticker (str): The ticker symbol of the stock or benchmark.
        start (str): Start date in the format 'YYYY-MM-DD'.
        end (str): End date in the format 'YYYY-MM-DD'.
        tipo (str): The type of data being fetched, e.g., 'Titolo' (stock) or 'Benchmark'.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    while True:
        try:
            BOLD = "\033[1m"
            END = "\033[0m"
            print(f"\nDownloading data for {tipo} '{ticker}'...\n")
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                print(f"\nError: No data found for the {tipo.lower()} '{ticker}'.")
                ticker = input(f"{BOLD} >>> {END} Please enter a valid ticker for the {tipo.lower()}:").strip()
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = data.columns.str.lower().str.replace(' ', '_', regex=False)
            if 'close' not in data.columns:
                print(f"\nError: The 'close' column is missing in the downloaded data for the {tipo.lower()}.")
                ticker = input(f"{BOLD} >>> {END} Please enter a valid ticker for the {tipo.lower()}:").strip()
                continue
            data = data.asfreq('B').ffill()
            return data
        except Exception as e:
            print(f"\n{BOLD}ERROR:{END} while downloading data for the {tipo.lower()} '{ticker}': {e}")
            ticker = input(f"{BOLD} >>> {END} Please enter a valid ticker for the {tipo.lower()}: ").strip()
            # If left empty during re-entry and the type is 'benchmark', return None
            if not ticker and tipo.lower() == 'benchmark':
                print(f"{BOLD}INFO:{END} No input provided. Using the default benchmark.")
                return None

# Differentiated functions for stock and benchmark data retrieval
def get_stock_data(ticker, start, end):
    """
    Fetches stock data for the given ticker and date range.

    Parameters:
        ticker (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing stock data.
    """
    return get_data(ticker, start, end, tipo='Titolo')

def get_benchmark_data(benchmark, start, end):
    """
    Fetches benchmark data for the given ticker and date range.

    Parameters:
        benchmark (str): Benchmark ticker symbol.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing benchmark data.
    """
    return get_data(benchmark, start, end, tipo='Benchmark')



