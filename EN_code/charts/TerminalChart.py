import plotly.graph_objects as go
import statsmodels.api as sm
import plotext as plt


# GRAPHS DIRECTLY FROM TERMINAL
def plot_histogram(data, ticker):
    if 'daily_returns' not in data.columns:
        print("Error: The 'daily_returns' column is not present in the data.")
        return
    daily_returns = data['daily_returns'].dropna()
    plt.clear_data()
    plt.theme('dark')
    plt.color("cyan")
    plt.edge_color("white")
    plt.hist(daily_returns, bins=50, label='Daily Returns')
    plt.title(f'Distribution of Daily Returns for {ticker}')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')


def plot_data(data, ticker):
    if data is None or data.empty:
        print(f"Error: The DataFrame for the ticker '{ticker}' is empty or invalid.")
        return
    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for '{ticker}'.")
        return
    data = data.dropna(subset=['close'])
    if data.empty:
        print(f"Error: Insufficient data to generate the chart for '{ticker}'.")
        return
    average_price = data['close'].mean()
    max_price = data['close'].max()
    min_price = data['close'].min()
    dates = data.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Historical Data with Average, Maximum, and Minimum Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.plot(dates, data['close'], label='Closing Price', color='cyan')
    plt.plot(dates, [average_price] * len(data), label='Average Price', color='yellow', linestyle='--')
    plt.plot(dates, [max_price] * len(data), label='Maximum Price', color='lime', linestyle=':')
    plt.plot(dates, [min_price] * len(data), label='Minimum Price', color='magenta', linestyle=':')
    plt.date_form('Y-m')
    return plt


def perform_seasonal_decomposition_plotext(data, ticker):
    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return
    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')
    observed_dates = decomposition.observed.index.strftime('01/%m/%Y').tolist()
    trend_dates = decomposition.trend.index.strftime('01/%m/%Y').tolist()
    seasonal_dates = decomposition.seasonal.index.strftime('01/%m/%Y').tolist()
    resid_dates = decomposition.resid.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Original Series (Observed in Cyan)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.plot(observed_dates, decomposition.observed, color='cyan')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Trend (Trend in Yellow)')
    plt.xlabel('Date')
    plt.ylabel('Trend')
    plt.plot(trend_dates, decomposition.trend, color='yellow')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Seasonality (Seasonality in Magenta)')
    plt.xlabel('Date')
    plt.ylabel('Seasonality')
    plt.plot(seasonal_dates, decomposition.seasonal, color='magenta')
    plt.show()
    print(" ")
    print(" ")
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Residual (Residual in Red)')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.plot(resid_dates, decomposition.resid, color='red')
    plt.show()
    print(" ")
    print(" ")

    return decomposition



def plot_daily_returns_plotext(data, ticker):
    if 'daily_returns' not in data.columns:
        print("Error: 'daily_returns' not calculated, unable to create the graph.")
        return
    daily_returns = data['daily_returns'].dropna()
    dates = daily_returns.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.plot(dates, daily_returns, label='Daily Returns', color='cyan')
    plt.show()


def plot_bollinger_bands_plotext(data, ticker):
    required_columns = ['close', 'bollinger_upper', 'bollinger_lower']
    for col in required_columns:
        if col not in data.columns:
            print(f"Error: '{col}' not present in the data, unable to create the graph.")
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
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.plot(dates, close, color='cyan')
    plt.plot(dates, bollinger_upper, color='magenta')
    plt.plot(dates, bollinger_lower, color='green')



def plot_rsi_plotext(data, ticker):
    if 'rsi' not in data.columns:
        print("Error: 'rsi' not calculated, unable to create the graph.")
        return
    rsi = data['rsi'].dropna()
    dates = rsi.index.strftime('01/%m/%Y').tolist()
    plt.clear_data()
    plt.theme('dark')
    plt.title(f'{ticker} - RSI (Relative Strength Index)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.plot(dates, rsi, label='RSI', color='cyan')
    plt.hline(30, color='green')
    plt.hline(70, color='red')
    plt.show()
    print("\nLegend:")
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
