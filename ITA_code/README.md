# FinancialStatsPython

FinancialStatsPython è un progetto Python progettato per eseguire analisi finanziarie dettagliate su azioni e indici di mercato. Questo script automatizza la raccolta dei dati, l'analisi del rischio, la generazione di grafici interattivi e la previsione dei prezzi, combinando tecniche statistiche avanzate con l'intelligenza artificiale. L'obiettivo è fornire uno strumento completo per la valutazione delle performance di mercato e la gestione del rischio.

---

## **Funzionalità principali**

### 1. Raccolta Dati
- Recupero automatico dei dati storici da Yahoo Finance.
- Calcolo di metriche chiave:
  - Prezzo medio.
  - Volumi.
  - Rendimenti giornalieri.

### 2. Analisi del Rischio e della Volatilità
- Calcolo di indicatori come:
  - **Volatilità.**
  - **Value at Risk (VaR)** e **Expected Shortfall (CVaR).**
  - **Sharpe Ratio** e altri.
- Analisi del massimo drawdown per valutare le maggiori perdite durante il periodo analizzato.

### 3. Modelli di Previsione
- Utilizzo dei modelli **ARIMA** e **GARCH** per la previsione dei prezzi e della volatilà.
- Implementazione di un modello AI (**LSTM**) per previsioni avanzate dei prezzi futuri.

### 4. Analisi Benchmark
- Confronto del titolo con benchmark di riferimento, incluso il calcolo di beta e correlazione.
- Valutazione della cointegrazione con il benchmark per analisi a lungo termine.

### 5. Visualizzazioni Interattive
- Generazione di grafici interattivi con **Plotly** e rappresentazioni dirette in terminale.
- **Decomposizione stagionale** per identificare trend, stagionalità e residui.

### 6. Dashboard HTML
- Creazione automatizzata di dashboard interattive per esplorare risultati e grafici.

---

## **Tecnologie Utilizzate**

### Librerie Python
- `pandas==2.2.3`
- `plotly==5.24.1`
- `plotext==5.3.2`
- `yfinance==0.2.51`
- `numpy==2.0.2`
- `scipy==1.15.0`
- `jinja2==3.1.5`
- `statsmodels==0.14.4`
- `tensorflow==2.18.0`
- `absl-py==2.1.0`
- `deep-translator==1.11.4`
- `matplotlib==3.10.0`
- `requests==2.32.3`
- `arch==7.2.0`
- `scikit-learn==1.6.1`
- `keras==3.8.0`

### Fonte dei Dati
- Yahoo Finance API.

### Modelli
- **ARIMA** per la modellazione autoregressiva dei prezzi.
- **GARCH** per la stima della volatilà condizionata.
- **LSTM** per previsioni basate su reti neurali ricorrenti.

---

## **Disclaimer**
Questo progetto è a scopo esclusivamente educativo e non costituisce consulenza finanziaria. L'autore declina esplicitamente qualsiasi responsabilità per errori, omissioni o inesattezze nelle analisi e nelle informazioni fornite. L'utilizzo di questi dati è a rischio esclusivo dell'utente. L'autore non garantisce l'accuratezza, la completezza o l'affidabilità delle informazioni presentate, né l'assenza di malfunzionamenti nello script utilizzato. Inoltre, l'autore non sarà ritenuto responsabile per eventuali danni diretti, indiretti, incidentali o consequenziali derivanti dall'uso o dall'affidamento su queste analisi. Gli utenti sono invitati a verificare e validare indipendentemente le informazioni prima di prendere decisioni basate su di esse.
