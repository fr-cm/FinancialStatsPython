import plotly.graph_objects as go
import statsmodels.api as sm
import plotext as plt


# GRAFICI DIRETTAMENTE DA TERMINALE
def plot_histogram(data, ticker):
    if 'daily_returns' not in data.columns:
        print("Errore: La colonna 'daily_returns' non è presente nei dati.")
        return
    daily_returns = data['daily_returns'].dropna()
    plt.clear_data()
    plt.theme('dark')
    plt.color("cyan")
    plt.edge_color("white")
    plt.hist(daily_returns, bins=50, label='Rendimenti Giornalieri')
    plt.title(f'Distribuzione dei Rendimenti Giornalieri di {ticker}')
    plt.xlabel('Rendimento Giornaliero')
    plt.ylabel('Frequenza')


def plot_data(data, ticker):
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
    dates = data.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Serie Storica con Prezzo Medio, Massimo e Minimo')
    plt.xlabel('Data')
    plt.ylabel('Prezzo di Chiusura')
    plt.plot(dates, data['close'], label='Prezzo di Chiusura', color='cyan')
    plt.plot(dates, [average_price] * len(data), label='Prezzo Medio', color='yellow', linestyle='--')
    plt.plot(dates, [max_price] * len(data), label='Prezzo Massimo', color='lime', linestyle=':')
    plt.plot(dates, [min_price] * len(data), label='Prezzo Minimo', color='magenta', linestyle=':')
    plt.date_form('Y-m')
    return plt


def perform_seasonal_decomposition_plotext(data, ticker):
    if 'close' not in data.columns:
        print(f"Errore: La colonna 'close' non è presente nei dati del titolo '{ticker}'.")
        return
    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')
    observed_dates = decomposition.observed.index.strftime('01/%m/%Y').tolist()
    trend_dates = decomposition.trend.index.strftime('01/%m/%Y').tolist()
    seasonal_dates = decomposition.seasonal.index.strftime('01/%m/%Y').tolist()
    resid_dates = decomposition.resid.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Serie Originale (Osservato in Cyan)')
    plt.xlabel('Data')
    plt.ylabel('Prezzo di Chiusura')
    plt.plot(observed_dates, decomposition.observed, color='cyan')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Trend (Trend in Yellow)')
    plt.xlabel('Data')
    plt.ylabel('Trend')
    plt.plot(trend_dates, decomposition.trend, color='yellow')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Stagionalità (Stagionalità in Magenta)')
    plt.xlabel('Data')
    plt.ylabel('Stagionalità')
    plt.plot(seasonal_dates, decomposition.seasonal, color='magenta')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Residuo (Residuo in Red)')
    plt.xlabel('Data')
    plt.ylabel('Residuo')
    plt.plot(resid_dates, decomposition.resid, color='red')
    plt.show()
    print(" ")
    print(" ")

    return decomposition


def plot_daily_returns_plotext(data, ticker):
    if 'daily_returns' not in data.columns:
        print("Errore: 'daily_returns' non calcolati, impossibile creare il grafico.")
        return
    daily_returns = data['daily_returns'].dropna()
    dates = daily_returns.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Rendimenti Giornalieri')
    plt.xlabel('Data')
    plt.ylabel('Rendimento Giornaliero')
    plt.plot(dates, daily_returns, label='Rendimenti Giornalieri', color='cyan')
    plt.show()


def plot_bollinger_bands_plotext(data, ticker):
    required_columns = ['close', 'bollinger_upper', 'bollinger_lower']
    for col in required_columns:
        if col not in data.columns:
            print(f"Errore: '{col}' non presente nei dati, impossibile creare il grafico.")
            return
    close = data['close'].dropna()
    bollinger_upper = data['bollinger_upper'].dropna()
    bollinger_lower = data['bollinger_lower'].dropna()
    common_index = close.index.intersection(bollinger_upper.index).intersection(bollinger_lower.index)
    close = close.loc[common_index]
    bollinger_upper = bollinger_upper.loc[common_index]
    bollinger_lower = bollinger_lower.loc[common_index]
    dates = close.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Bollinger Bands')
    plt.xlabel('Data')
    plt.ylabel('Prezzo di Chiusura')
    plt.plot(dates, close, color='cyan')
    plt.plot(dates, bollinger_upper, color='magenta')
    plt.plot(dates, bollinger_lower, color='green')



def plot_rsi_plotext(data, ticker):
    if 'rsi' not in data.columns:
        print("Errore: 'rsi' non calcolato, impossibile creare il grafico.")
        return
    rsi = data['rsi'].dropna()
    dates = rsi.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - RSI (Relative Strength Index)')
    plt.xlabel('Data')
    plt.ylabel('RSI')
    plt.plot(dates, rsi, label='RSI', color='cyan')
    plt.hline(30, color='green')
    plt.hline(70, color='red')
    plt.show()
    print("\nLegenda:")
    print(" - Cyan: RSI")
    print(" - Green: RSI=30")
    print(" - Red: RSI=70")


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