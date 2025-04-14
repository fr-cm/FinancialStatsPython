import os
import sys
import subprocess

libraries = [
    "pandas==2.2.3",
    "plotly==5.24.1",
    "plotext==5.3.2",
    "yfinance==0.2.55",
    "numpy==2.0.2",
    "scipy==1.15.0",
    "jinja2==3.1.5",
    "statsmodels==0.14.4",
    "tensorflow==2.18.0",
    "absl-py==2.1.0",
    "deep-translator==1.11.4",
    "matplotlib==3.10.0",
    "requests==2.32.3",
    "arch==7.2.0",
    "scikit-learn==1.6.1",
    "keras==3.8.0",

]

def install_library(lib):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"Libreria {lib} installata con successo.")
    except Exception as e:
        print(f"Errore durante l'installazione di {lib}: {e}")

def main():
    print("Inizio installazione delle librerie necessarie...")
    for lib in libraries:
        install_library(lib)
    print("Installazione completata!")

if __name__ == "__main__":
    main()
