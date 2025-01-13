import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import statsmodels.api as sm



#  GRAFICI CON PLOTLY
def plot_histogram_plotly(data, ticker):

    if 'daily_returns' not in data.columns:
        print("Errore: La colonna 'daily_returns' non è presente nei dati.")
        return

    fig = px.histogram(
        data['daily_returns'].dropna(),
        nbins=50,
        title=f'Distribuzione dei Rendimenti Giornalieri di {ticker}',
        labels={'value': 'Rendimento Giornaliero', 'count': 'Frequenza'},
        opacity=0.75,
        color_discrete_sequence=['#1f77b4']
    )

    fig.update_layout(
        xaxis_title='Rendimento Giornaliero',
        yaxis_title='Frequenza',
        template='plotly_white',
        legend_title='',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )


    mean = data['daily_returns'].mean()
    std = data['daily_returns'].std()
    fig.add_vline(x=mean, line=dict(color='#ff7f0e', dash='dash'), annotation_text='Media',
                  annotation_position="top left")
    fig.add_vline(x=mean + std, line=dict(color='#2ca02c', dash='dot'), annotation_text='+1 SD',
                  annotation_position="top left")
    fig.add_vline(x=mean - std, line=dict(color='#d62728', dash='dot'), annotation_text='-1 SD',
                  annotation_position="top left")

    return fig

def plot_correlation_plotly(data, benchmark_data, ticker, benchmark):

    if data is None or benchmark_data is None:
        print("Errore: Dati mancanti per la correlazione.")
        return None

    merged = pd.DataFrame({
        'Stock': data['daily_returns'],
        'Benchmark': benchmark_data['daily_returns']
    }).dropna()

    if merged.empty:
        print("Errore: Nessun dato sovrapponibile per la correlazione.")
        return None

    fig = px.scatter(merged, x='Benchmark', y='Stock',
                     trendline='ols',
                     title=f'Correlazione tra {ticker} e {benchmark}',
                     labels={'Benchmark': f'Rendimenti {benchmark}', 'Stock': f'Rendimenti {ticker}'})

    fig.update_layout(template='plotly_white')

    return fig

def plot_data_plotly(data, ticker):

    if data is None or data.empty:
        print(f"Errore: Il DataFrame del titolo '{ticker}' è vuoto o non valido.")
        return

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    data = data.dropna(subset=['close'])
    if data.empty:
        print(f"Errore: Dati insufficienti per generare il grafico per '{ticker}'.")
        return

    average_price = data['close'].mean()
    max_price = data['close'].max()
    min_price = data['close'].min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines',
                             name=f'{ticker} - Prezzo di Chiusura',
                             line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=data.index, y=[average_price] * len(data), mode='lines',
                             name='Prezzo Medio',
                             line=dict(color='orange', dash='dash')))

    fig.add_trace(go.Scatter(x=data.index, y=[max_price] * len(data), mode='lines',
                             name='Prezzo Massimo',
                             line=dict(color='green', dash='dot')))

    fig.add_trace(go.Scatter(x=data.index, y=[min_price] * len(data), mode='lines',
                             name='Prezzo Minimo',
                             line=dict(color='red', dash='dot')))

    fig.update_layout(
        title=f'{ticker} - Serie Storica',
        xaxis_title='',
        yaxis_title='Prezzo',
        template='plotly_white',
        legend_title='',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5
        ))

    return fig


def plot_story_plotly(data, ticker):

    if data is None or data.empty:
        print(f"Errore: Il DataFrame del titolo '{ticker}' è vuoto o non valido.")
        return

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    data = data.dropna(subset=['close'])
    if data.empty:
        print(f"Errore: Dati insufficienti per generare il grafico per '{ticker}'.")
        return

    average_price = data['close'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines',
                             name=f'{ticker} - Prezzo di Chiusura',
                             line=dict(color='darkslateblue', width=2)))

    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='Prezzo',
        template='plotly_white',
        legend_title=' '
    )

    return fig



def perform_seasonal_decomposition_plotly(data, ticker):

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')


    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Serie Storica', 'Trend', 'Stagionalità', 'Residuo')
    )

    # Serie Originale
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index,
        y=decomposition.observed,
        mode='lines',
        name='Osservato',
        line=dict(color='#1f77b4')
    ), row=1, col=1)

    # Trend
    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend,
        mode='lines',
        name='Trend',
        line=dict(color='#ff7f0e')
    ), row=2, col=1)

    # Stagionalità
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Stagionalità',
        line=dict(color='#2ca02c')
    ), row=3, col=1)

    # Residuo
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residuo',
        line=dict(color='#d62728')
    ), row=4, col=1)

    fig.update_layout(
        height=900,
        width=1200,
        title_text=f'{ticker} - Decomposizione Stagionale',
        template='plotly_white',
        showlegend=False
    )

    fig.update_xaxes(title_text='', row=4, col=1)
    fig.update_yaxes(title_text='Prezzo', row=1, col=1)
    fig.update_yaxes(title_text='Trend', row=2, col=1)
    fig.update_yaxes(title_text='Stagionalità', row=3, col=1)
    fig.update_yaxes(title_text='Residuo', row=4, col=1)
    return fig

    return decomposition

def plot_original_series(data, ticker):

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index,
        y=decomposition.observed,
        mode='lines',
        name='Osservato',
        line=dict(color='#1f77b4')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Prezzo",
        template="plotly_white"
    )


    return fig

def plot_trend(data, ticker):

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend,
        mode='lines',
        name='Trend',
        line=dict(color='#ff7f0e')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Trend",
        template="plotly_white"
    )


    return fig

def plot_seasonality(data, ticker):
    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Stagionalità',
        line=dict(color='#2ca02c')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Stagionalità",
        template="plotly_white"
    )


    return fig

def plot_residual(data, ticker):
    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residuo',
        line=dict(color='#d62728')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Residuo",
        template="plotly_white"
    )

    return fig


def plot_daily_returns_plotly(data, ticker):

    if 'daily_returns' not in data.columns:
        print("Errore: 'daily_returns' non calcolati, impossibile creare il grafico.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['daily_returns'],
        mode='lines',
        name='Rendimenti Giornalieri',
        line=dict(color='#9467bd', width=2)
    ))

    fig.update_layout(
        title=f"{ticker} - Rendimenti Giornalieri",
        xaxis_title='Data',
        yaxis_title='Rendimento (%)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    return fig


def plot_rsi_plotly(data, ticker):

    if 'rsi' not in data.columns:
        print("Errore: 'rsi' non calcolato, impossibile creare il grafico.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['rsi'],
        mode='lines',
        name='RSI',
        line=dict(color='#17becf', width=2)
    ))


    fig.add_hline(y=30, line=dict(color='#2ca02c', dash='dash'), annotation_text='RSI=30',
                  annotation_position="bottom left")
    fig.add_hline(y=70, line=dict(color='#d62728', dash='dash'), annotation_text='RSI=70',
                  annotation_position="top left")

    fig.update_layout(
        title=f"{ticker} - RSI (Relative Strength Index)",
        xaxis_title='',
        yaxis_title='RSI',
        template='plotly_white',
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    return fig


def plot_bollinger_bands_plotly(data, ticker):

    required_columns = ['close', 'bollinger_upper', 'bollinger_lower']
    for col in required_columns:
        if col not in data.columns:
            print(f"Errore: '{col}' non presente nei dati, impossibile creare il grafico.")
            return

    fig = go.Figure()

    # Prezzo di Chiusura
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Close',
        line=dict(color='#1f77b4', width=2)
    ))

    # Bollinger Upper
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bollinger_upper'],
        mode='lines',
        name='Bollinger Upper',
        line=dict(color='#ff7f0e', dash='dash'),
        opacity=0.6
    ))

    # Bollinger Lower
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bollinger_lower'],
        mode='lines',
        name='Bollinger Lower',
        line=dict(color='#2ca02c', dash='dash'),
        opacity=0.6
    ))

    # Layout
    fig.update_layout(
        title=f"{ticker} - Bollinger Bands",
        xaxis_title='',
        yaxis_title='Prezzo',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    return fig



def plot_adfuller_test(adf_result, ticker):

    test_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    fig = go.Figure()

    for key, value in critical_values.items():
        fig.add_shape(
            type="line",
            x0=0,
            y0=value,
            x1=1,
            y1=value,
            line=dict(color="red", dash="dash"),
        )
        fig.add_annotation(
            x=1.02,
            y=value,
            xref="x",
            yref="y",
            text=f"{key} Crit. Val",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(color="red")
        )

    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[test_stat],
        mode='markers',
        marker=dict(color='blue', size=12),
        name='Test Statistic'
    ))


    fig.add_annotation(
        x=0.5,
        y=test_stat,
        xref="x",
        yref="y",
        text=f"ADF Stat: {test_stat:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    fig.update_layout(
        title=f"",
        xaxis=dict(showticklabels=False, zeroline=False, range=[0, 1]),
        yaxis_title="Test Statistic",
        showlegend=False,
        template='plotly_white',
        height=500
    )

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

    return fig

import pandas as pd
import plotly.graph_objects as go

def plot_moving_averages(data, ticker, sma_periods=[50, 200], ema_periods=[50, 200]):
    """
    Crea un grafico con SMA ed EMA sovrapposte al prezzo di chiusura.
    :param data: DataFrame con colonne 'close', 'sma_<period>' e 'ema_<period>'.
    :param ticker: Nome del titolo.
    :param sma_periods: Liste di periodi per SMA.
    :param ema_periods: Liste di periodi per EMA.
    :return: Figura Plotly.
    """
    fig = go.Figure()

    # Prezzo di chiusura
    fig.add_trace(go.Scatter(
        x=data.index, y=data['close'],
        mode='lines', name='Prezzo Chiusura',
        line=dict(color='blue', width=2)
    ))

    # Medie mobili semplici (SMA)
    for period in sma_periods:
        col_name = f'sma_{period}'
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                mode='lines', name=f'SMA {period}',
                line=dict(dash='dash')
            ))

    # Medie mobili esponenziali (EMA)
    for period in ema_periods:
        col_name = f'ema_{period}'
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                mode='lines', name=f'EMA {period}',
                line=dict(dash='dot')
            ))

    fig.update_layout(
        title=f'{ticker} - Medie Mobili',
        xaxis_title='',
        yaxis_title='Prezzo',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def plot_average_volume(data, ticker, period=50):

    col_name = f'avg_volume_{period}'
    if col_name not in data.columns:
        raise ValueError(f"Colonna '{col_name}' non trovata nel DataFrame.")

    fig = go.Figure()

    # Volume giornaliero
    fig.add_trace(go.Bar(
        x=data.index, y=data['volume'],
        name='Volume Giornaliero',
        marker_color='lightblue'
    ))

    # Volume medio
    fig.add_trace(go.Scatter(
        x=data.index, y=data[col_name],
        mode='lines', name=f'Volume Medio {period}',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title=f'{ticker} - Volume e Media {period} Giorni',
        xaxis_title='',
        yaxis_title='Volume',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_momentum(data, ticker, period=14):

    col_name = f'momentum_{period}'
    if col_name not in data.columns:
        raise ValueError(f"Colonna '{col_name}' non trovata nel DataFrame.")

    fig = go.Figure()

    # Momentum
    fig.add_trace(go.Scatter(
        x=data.index, y=data[col_name],
        mode='lines', name=f'Momentum {period}',
        line=dict(color='purple')
    ))

    # Linea dello zero
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), annotation_text='Zero',
                  annotation_position="top left")

    fig.update_layout(
        title=f'{ticker} - Momentum {period} Giorni',
        xaxis_title='',
        yaxis_title='Momentum',
        template='plotly_white',
        legend = dict(
            orientation="h",
            yanchor="top",
            y=1.02,
           xanchor="right",
           x=1
    )
    )
    return fig




def plot_stationarity(data, column='close', diff_column='diff_close'):
    fig = go.Figure()

    # Grafico della serie originale
    fig.add_trace(go.Scatter(
        x=data.index, y=data[column], mode='lines', name='Serie Originale',
        line=dict(color='blue')
    ))

    # Grafico della serie differenziata
    if diff_column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[diff_column], mode='lines', name='Serie Differenziata',
            line=dict(color='orange')
        ))

    fig.update_layout(
        title="Stazionarietà: Serie Originale e Differenziata",
        xaxis_title="",
        yaxis_title="Valore",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=0.5)
    )

    return fig


# grafico ARIMA
def plot_arima_predictions(data, arima_model, column='close'):
    predictions = arima_model.predict(start=data.index[0], end=data.index[-1])

    fig = go.Figure()

    # Serie reale
    fig.add_trace(go.Scatter(
        x=data.index, y=data[column], mode='lines', name='Reale',
        line=dict(color='blue')
    ))

    # Previsioni ARIMA
    fig.add_trace(go.Scatter(
        x=data.index, y=predictions, mode='lines', name='Previsioni ARIMA',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="Confronto Serie Reale e Previsioni ARIMA",
        xaxis_title="",
        yaxis_title="Valore",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=0.5
    ))

    return fig


#  grafico GARCH
def plot_garch_volatility(data, garch_model):
    conditional_volatility = garch_model.conditional_volatility

    fig = go.Figure()

    # Serie di volatilità
    fig.add_trace(go.Scatter(
        x=data.index, y=conditional_volatility, mode='lines', name='Volatilità Condizionata',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title="Volatilità Condizionata Stimata (GARCH)",
        xaxis_title="",
        yaxis_title="Volatilità",
        template="plotly_white",

    )

    return fig
def plot_var_analysis(data, var_value, var_h, cvar_95, confidence_level=0.95):

    if 'daily_returns' not in data.columns:
        print("Errore: La colonna 'daily_returns' non è presente nei dati.")
        return

    daily_returns = data['daily_returns'].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_returns,
        nbinsx=50,
        name='Rendimenti Giornalieri',
        opacity=0.75,
        marker_color='#1f77b4'
    ))

    # Linea del VaR
    fig.add_vline(x=var_value, line=dict(color='red', dash='dash'),
                  annotation_text=f"VaR ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    # Linea del VaR storico
    fig.add_vline(x=var_h, line=dict(color='orange', dash='dot'),
                  annotation_text=f"VaR Storico ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    # Linea del CVaR
    fig.add_vline(x=cvar_95, line=dict(color='green', dash='solid'),
                  annotation_text=f"CVaR ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    fig.update_layout(
        title="Analisi di VaR, VaR Storico e CVaR",
        xaxis_title="Rendimenti Giornalieri",
        yaxis_title="Frequenza",
        template="plotly_white",
        legend_title="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=0.5
    )
    )

    return fig


import plotly.graph_objects as go


def plot_all_models_forecast(data, results, split_idx):
    """
    Plotta sullo stesso grafico:
      - La serie storica (train + test)
      - I valori reali del test set (in verde)
      - Le predizioni sul test set di ogni modello
      - Le previsioni future di ogni modello

    Parametri:
    ----------
    data : pd.DataFrame
        Deve contenere almeno la colonna 'close'. L'indice è di tipo datetime.
    results : dict
        Dizionario con le chiavi dei modelli, p.es. "LSTM", "CNN+LSTM", "Transformer".
        Ciascuna chiave deve mappare a un dict con:
          {
            'y_test_rescaled': np.array,
            'test_pred_rescaled': np.array,
            'forecast_df': pd.DataFrame con colonna 'forecast',
            ...
          }
    split_idx : int
        Indice che separa la fine del train dall'inizio del test in 'data'.
        Usato per allineare correttamente il test set sul grafico.
    """

    if data is None or len(data) == 0:
        raise ValueError("DataFrame 'data' vuoto o None.")
    if 'close' not in data.columns:
        raise ValueError("Il DataFrame storico deve contenere la colonna 'close'.")
    if not isinstance(results, dict) or len(results) == 0:
        raise ValueError("Il parametro 'results' deve essere un dizionario non vuoto.")
    if split_idx is None or not isinstance(split_idx, int):
        raise ValueError("split_idx deve essere un intero valido.")
    if split_idx < 0 or split_idx >= len(data):
        raise ValueError("split_idx è fuori dai limiti dell'indice del DataFrame.")

    # 1) Creiamo la figura
    fig = go.Figure()

    # 2) Serie storica (train + test) in blu
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Prezzo Storico',
        line=dict(color='blue')
    ))

    # 3) Parte reale (solo test set) in verde
    first_model_key = next(iter(results))  # es. 'LSTM'
    y_test_rescaled = results[first_model_key]['y_test_rescaled']
    n_test = len(y_test_rescaled)

    test_index = data.index[split_idx + 1: split_idx + 1 + n_test]

    fig.add_trace(go.Scatter(
        x=test_index,
        y=y_test_rescaled.flatten(),
        mode='lines',
        name='Reale (test set)',
        line=dict(color='green')
    ))

    # 4) Definiamo colori e stili per i 3 modelli
    model_colors = {
        "LSTM": "red",
        "CNN+LSTM": "orange",
        "Transformer": "purple"
    }
    # Per le previsioni future, usa linee tratteggiate
    dash_styles = {
        "LSTM": "solid",
        "CNN+LSTM": "dot",
        "Transformer": "dash"
    }

    # 5) Aggiunge le curve di predizione test e forecast future per ogni modello
    for model_name, model_dict in results.items():
        # (a) Predizioni su test set
        test_pred_rescaled = model_dict['test_pred_rescaled']
        if test_pred_rescaled is None:
            continue
        if len(test_pred_rescaled) != n_test:
            raise ValueError(f"Dimensione test_pred_rescaled di {model_name} non coincide con y_test_rescaled.")

        fig.add_trace(go.Scatter(
            x=test_index,
            y=test_pred_rescaled.flatten(),
            mode='lines+markers',
            name=f'Pred. Test ({model_name})',
            line=dict(color=model_colors.get(model_name, 'gray')),
            showlegend=True
        ))

        # (b) Previsioni future
        forecast_df = model_dict['forecast_df']
        if forecast_df is not None and 'forecast' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines+markers',
                name=f'Future ({model_name})',
                line=dict(color=model_colors.get(model_name, 'gray'),
                          dash=dash_styles.get(model_name, 'dot')),
                showlegend=True
            ))

    # 6) Layout
    fig.update_layout(
        title="Confronto Previsioni Multi-Modello",
        xaxis_title="",
        yaxis_title="Prezzo",
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor = "rgba(0,0,0,0)"
        )
    )

    return fig


def plot_seasonality_and_residual(data, ticker):

    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()

    # stagionalità
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Stagionalità',
        line=dict(color='#2ca02c')
    ))

    #  residuo
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residuo',
        line=dict(color='#d62728')
    ))

    fig.update_layout(
        title=f"Stagionalità e Residuo per {ticker}",
        xaxis_title="",
        yaxis_title="Valore",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    return fig
