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


# Aggiorna le date predefinite in base ai dati disponibili
def get_default_dates(ticker):
    """
    Recupera la prima e l'ultima data disponibile per un ticker su Yahoo Finance.
    """
    try:
        # Scarica tutti i dati disponibili
        data = yf.download(ticker, period="max")
        if not data.empty:
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            return start_date, end_date
        else:
            raise ValueError("Dati storici non disponibili per il ticker.")
    except Exception as e:
        print(f"Errore nel recuperare le date predefinite: {e}")
        return None, None

def get_valid_date(prompt, default):
    """
    input data nel formato YYYY-MM-DD.
    Se l'input è vuoto, restituisce la data predefinita.
    Se l'input non è valido, richiede nuovamente l'input.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            print(f"     : Nessuna data inserita. Utilizzo la data predefinita: {default}")
            return default
        # Prova diversi formati di data
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d'):
            try:
                parsed_date = datetime.strptime(user_input, fmt)
                formatted_date = parsed_date.strftime('%Y-%m-%d')
                print(f"     : Data inserita valida: {formatted_date}")
                return formatted_date
            except ValueError:
                continue
        # Se nessun formato ha funzionato, informa l'utente e riprova
        print("Errore: Inserisci una data corretta nel formato YYYY-MM-DD (es. 2023-01-31).")


########################################################################################################
########################################################################################################
#                                            SLOW_PRINT                                                #
########################################################################################################
########################################################################################################
def slow_print(text, delay=0.002):
    """
    Stampa il testo carattere per carattere con un ritardo.
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')  # Aggiunge una nuova linea alla fine


def slow_print_line(text, delay=0.002):
    """
    Stampa il testo riga per riga con un ritardo tra le righe.
    """
    print(text)
    time.sleep(delay)




def spinner(messaggio, stop_event):
    """
    animazione di caricaricamento
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
    Genera una dashboard HTML il template da GitHub di Interfaccia ,
    la salva nella cartella Download e la apre nel browser.
    """
    try:
        # URL Template
        template_url = "https://raw.githubusercontent.com/fr-cm/interfaccia/main/Code/Template_interfaccia.html"

        # Scarica il template
        response = requests.get(template_url)
        if response.status_code != 200:
            logging.error(f"Impossibile scaricare il template. Stato HTTP: {response.status_code}")
            print(f"Errore: Impossibile scaricare il template dal link fornito. Stato HTTP: {response.status_code}")
            return

        template_content = response.text

        # Crea un oggetto Template
        template = Template(template_content)
        # Renderizza il template con i dati, passandoli come 'data'
        rendered_html = template.render(data=data_ds)
        # Trova la cartella Download in modo cross-platform
        home = Path.home()
        download_dir = home / "Downloads"

        if not download_dir.exists():
            logging.warning(f"La cartella Downloads non esiste: {download_dir}. Verrà utilizzata la home directory.")
            download_dir = home

        # percorso di output
        output_filename = f"{ticker}_dashboard.html"
        output_path = download_dir / output_filename

        # Scrive l'HTML renderizzato nel file di output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        logging.info(f"Dashboard generata con successo: {output_path}")
        print(f"Dashboard generata con successo: {output_path}")

        # Apre automaticamente la dashboard nel browser
        try:
            file_url = output_path.resolve().as_uri()
            webbrowser.open(file_url)
            logging.info(f"Dashboard aperta automaticamente nel browser: {file_url}")
            print(f"Dashboard aperta automaticamente nel browser: {file_url}")
        except Exception as e:
            logging.error(f"Errore nell'apertura della dashboard nel browser: {e}")
            print(f"Errore nell'apertura della dashboard nel browser: {e}")

    except Exception as e:
        logging.error(f"Errore durante la generazione della dashboard: {e}")
        print(f"Errore durante la generazione della dashboard: {e}")


########################################################################################################
########################################################################################################
#                                            GET_STOCK_INFO                                            #
########################################################################################################
########################################################################################################

def get_stock_info(ticker):
    """
    Recupera le informazioni generali del titolo come capitalizzazione di mercato, settore, industria, ecc.
    Se la descrizione non è disponibile, utilizza un testo predefinito.
    """
    default_description = (
      "Le analisi forniscono una panoramica completa delle performance e dei rischi di un titolo, combinando metodi statistici, machine learning e grafici avanzati. Partono dal confronto tra prezzo medio storico e prezzo recente, utile per identificare tendenze. Si calcolano il momentum, per valutare la forza dei trend, e le medie mobili esponenziali, per individuare tendenze a breve e lungo termine. La volatilità annualizzata e il massimo drawdown misurano il rischio e la stabilità del titolo, mentre il VaR e il CVaR quantificano le perdite potenziali. Gli indici Sharpe, Treynor, Sortino e Calmar valutano il rendimento rispetto al rischio. Beta e alpha analizzano la sensibilità e la sovra/sottoperformance rispetto al benchmark, mentre correlazione e cointegrazione esplorano legami di lungo termine. Modelli ARIMA e GARCH prevedono prezzi futuri e stimano volatilità, supportati da previsioni con intelligenza artificiale. Grafici interattivi e statici completano l'analisi, offrendo strumenti chiave per decisioni di investimento informate.")
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        # Recupera la descrizione
        description = info.get('longBusinessSummary', '').strip()
        logging.debug(f"Descrizione originale per {ticker}: {description}")
        if description and description.lower() not in ['n/a', 'not available', '']:
            try:
                # Traduzione della descrizione in italiano
                description_it = translator.translate(description)
                logging.debug(f"Descrizione tradotta: {description_it}")
            except Exception as e:
                logging.error(f"Errore nella traduzione della descrizione per il ticker {ticker}: {e}")
                description_it = default_description
        else:
            logging.warning(f"Nessuna descrizione disponibile per il ticker {ticker}. Utilizzo del testo predefinito.")
            description_it = default_description

            # Recupera il nome lungo, utilizza il ticker come fallback
        long_name = info.get('longName', '').strip()
        if not long_name or long_name.lower() in ['n/a', 'not available', '','N/A']:
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
            'description': description_it,
        }
        return general_info
    except Exception as e:
        logging.error(f"Errore nel recuperare le informazioni generali per il ticker {ticker}: {e}")
        return {}


########################################################################################################
########################################################################################################
#                                            GET_LOGO_URL                                              #
########################################################################################################
########################################################################################################

def get_logo_url(ticker):
    """
    Recupera l'URL del logo dell'azienda associata al ticker fornito.
    Imposta una larghezza minima di 400px e utilizza un'immagine predefinita se il logo non è disponibile.
    """
    default_logo_url = "https://raw.githubusercontent.com/fr-cm/interfaccia/refs/heads/main/Tutorial/img/interfaccia_1.png"
    try:
        ticker_info = yf.Ticker(ticker).info
        website = ticker_info.get('website')
        if website:
            # Estrae il dominio dal sito web
            parsed_url = urlparse(website)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            logo_url = f"https://logo.clearbit.com/{domain}"

            # Verifica se il logo esiste
            response = requests.get(logo_url)
            if response.status_code == 200:
                return logo_url
            else:
                logging.warning(f"Logo non trovato per il dominio {domain}. Utilizzo logo predefinito.")
                return default_logo_url
        else:
            logging.warning(f"Sito web non trovato per il ticker {ticker}. Utilizzo logo predefinito.")
            return default_logo_url
    except Exception as e:
        logging.error(f"Errore nel recuperare il logo per il ticker {ticker}: {e}")
        return default_logo_url



########################################################################################################
########################################################################################################
#                                            GET_STOCK_DATA                                            #
########################################################################################################
########################################################################################################

def get_data(ticker, start, end, tipo='Titolo'):
    while True:
        try:
            BOLD = "\033[1m"
            END = "\033[0m"
            print(f"\nScaricando i dati per {tipo} '{ticker}'...\n")
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                print(f"\nErrore: Nessun dato trovato per il {tipo.lower()} '{ticker}'.")
                ticker = input(f"{BOLD} >>> {END} Inserisci un ticker valido per il {tipo.lower()}:").strip()
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = data.columns.str.lower().str.replace(' ', '_', regex=False)
            if 'close' not in data.columns:
                print(f"\nErrore: La colonna 'close' non è presente nei dati scaricati per il {tipo.lower()}.")
                ticker = input(f"{BOLD} >>> {END} Inserisci un ticker valido per il {tipo.lower()}:").strip()
                continue
            data = data.asfreq('B').ffill()
            return data
        except Exception as e:
            print(f"\n{BOLD}ERRORE:{END} durante il download dei dati per il {tipo.lower()} '{ticker}': {e}")
            ticker = input(f"{BOLD} >>> {END} Inserisci un ticker valido per il {tipo.lower()}: ").strip()
            # Se si lascia vuoto durante il reinserimento e il tipo è 'benchmark', ritorna None
            if not ticker and tipo.lower() == 'benchmark':
                print(f"{BOLD}INFO:{END} Nessun input fornito. Utilizzo del benchmark predefinito.")
                return None

# differrenzzazione per sapere che imput si riceve
def get_stock_data(ticker, start, end):
    return get_data(ticker, start, end, tipo='Titolo')

def get_benchmark_data(benchmark, start, end):
    return get_data(benchmark, start, end, tipo='Benchmark')


