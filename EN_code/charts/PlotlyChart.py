import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import statsmodels.api as sm


# GRAPHS WITH PLOTLY
def plot_histogram_plotly(data, ticker):

    if 'daily_returns' not in data.columns:
        print("Error: The 'daily_returns' column is not present in the data.")
        return

    fig = px.histogram(
        data['daily_returns'].dropna(),
        nbins=50,
        title=f'Distribution of Daily Returns for {ticker}',
        labels={'value': 'Daily Return', 'count': 'Frequency'},
        opacity=0.75,
        color_discrete_sequence=['#1f77b4']
    )

    fig.update_layout(
        xaxis_title='Daily Return',
        yaxis_title='Frequency',
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
    fig.add_vline(x=mean, line=dict(color='#ff7f0e', dash='dash'), annotation_text='Mean',
                  annotation_position="top left")
    fig.add_vline(x=mean + std, line=dict(color='#2ca02c', dash='dot'), annotation_text='+1 SD',
                  annotation_position="top left")
    fig.add_vline(x=mean - std, line=dict(color='#d62728', dash='dot'), annotation_text='-1 SD',
                  annotation_position="top left")

    return fig

def plot_correlation_plotly(data, benchmark_data, ticker, benchmark):

    if data is None or benchmark_data is None:
        print("Error: Missing data for correlation.")
        return None

    merged = pd.DataFrame({
        'Stock': data['daily_returns'],
        'Benchmark': benchmark_data['daily_returns']
    }).dropna()

    if merged.empty:
        print("Error: No overlapping data for correlation.")
        return None

    fig = px.scatter(merged, x='Benchmark', y='Stock',
                     trendline='ols',
                     title=f'Correlation between {ticker} and {benchmark}',
                     labels={'Benchmark': f'Returns {benchmark}', 'Stock': f'Returns {ticker}'})

    fig.update_layout(template='plotly_white')

    return fig

def plot_data_plotly(data, ticker):

    if data is None or data.empty:
        print(f"Error: The DataFrame for the ticker '{ticker}' is empty or invalid.")
        return

    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    data = data.dropna(subset=['close'])
    if data.empty:
        print(f"Error: Insufficient data to generate the graph for '{ticker}'.")
        return

    average_price = data['close'].mean()
    max_price = data['close'].max()
    min_price = data['close'].min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines',
                             name=f'{ticker} - Closing Price',
                             line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=data.index, y=[average_price] * len(data), mode='lines',
                             name='Average Price',
                             line=dict(color='orange', dash='dash')))

    fig.add_trace(go.Scatter(x=data.index, y=[max_price] * len(data), mode='lines',
                             name='Maximum Price',
                             line=dict(color='green', dash='dot')))

    fig.add_trace(go.Scatter(x=data.index, y=[min_price] * len(data), mode='lines',
                             name='Minimum Price',
                             line=dict(color='red', dash='dot')))

    fig.update_layout(
        title=f'{ticker} - Historical Data',
        xaxis_title='',
        yaxis_title='Price',
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
        print(f"Error: The DataFrame for the ticker '{ticker}' is empty or invalid.")
        return

    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    data = data.dropna(subset=['close'])
    if data.empty:
        print(f"Error: Insufficient data to generate the graph for '{ticker}'.")
        return

    average_price = data['close'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines',
                             name=f'{ticker} - Closing Price',
                             line=dict(color='darkslateblue', width=2)))

    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='Price',
        template='plotly_white',
        legend_title=' '
    )

    return fig



def perform_seasonal_decomposition_plotly(data, ticker):

    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')


    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Historical Data', 'Trend', 'Seasonality', 'Residual')
    )

    # Original Series
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index,
        y=decomposition.observed,
        mode='lines',
        name='Observed',
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

    # Seasonality
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonality',
        line=dict(color='#2ca02c')
    ), row=3, col=1)

    # Residual
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residual',
        line=dict(color='#d62728')
    ), row=4, col=1)

    fig.update_layout(
        height=900,
        width=1200,
        title_text=f'{ticker} - Seasonal Decomposition',
        template='plotly_white',
        showlegend=False
    )

    fig.update_xaxes(title_text='', row=4, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Trend', row=2, col=1)
    fig.update_yaxes(title_text='Seasonality', row=3, col=1)
    fig.update_yaxes(title_text='Residual', row=4, col=1)
    return fig

    return decomposition

def plot_original_series(data, ticker):

    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index,
        y=decomposition.observed,
        mode='lines',
        name='Observed',
        line=dict(color='#1f77b4')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Price",
        template="plotly_white"
    )


    return fig

def plot_trend(data, ticker):

    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
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
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonality',
        line=dict(color='#2ca02c')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Seasonality",
        template="plotly_white"
    )


    return fig

def plot_residual(data, ticker):
    if 'close' not in data.columns:
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residual',
        line=dict(color='#d62728')
    ))

    fig.update_layout(
        title=f"",
        xaxis_title="",
        yaxis_title="Residual",
        template="plotly_white"
    )

    return fig

def plot_daily_returns_plotly(data, ticker):

    if 'daily_returns' not in data.columns:
        print("Error: 'daily_returns' not calculated, unable to create the graph.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['daily_returns'],
        mode='lines',
        name='Daily Returns',
        line=dict(color='#9467bd', width=2)
    ))

    fig.update_layout(
        title=f"{ticker} - Daily Returns",
        xaxis_title='Date',
        yaxis_title='Return (%)',
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
        print("Error: 'rsi' not calculated, unable to create the graph.")
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
            print(f"Error: '{col}' not present in the data, unable to create the graph.")
            return

    fig = go.Figure()

    # Closing Price
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
        yaxis_title='Price',
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


def plot_moving_averages(data, ticker, sma_periods=[50, 200], ema_periods=[50, 200]):
    """
    Creates a graph with SMA and EMA overlaid on the closing price.
    :param data: DataFrame with 'close', 'sma_<period>', and 'ema_<period>' columns.
    :param ticker: Ticker symbol.
    :param sma_periods: List of periods for SMA.
    :param ema_periods: List of periods for EMA.
    :return: Plotly figure.
    """
    fig = go.Figure()

    # Closing price
    fig.add_trace(go.Scatter(
        x=data.index, y=data['close'],
        mode='lines', name='Closing Price',
        line=dict(color='blue', width=2)
    ))

    # Simple Moving Averages (SMA)
    for period in sma_periods:
        col_name = f'sma_{period}'
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                mode='lines', name=f'SMA {period}',
                line=dict(dash='dash')
            ))

    # Exponential Moving Averages (EMA)
    for period in ema_periods:
        col_name = f'ema_{period}'
        if col_name in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_name],
                mode='lines', name=f'EMA {period}',
                line=dict(dash='dot')
            ))

    fig.update_layout(
        title=f'{ticker} - Moving Averages',
        xaxis_title='',
        yaxis_title='Price',
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
        raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

    fig = go.Figure()

    # Daily volume
    fig.add_trace(go.Bar(
        x=data.index, y=data['volume'],
        name='Daily Volume',
        marker_color='lightblue'
    ))

    # Average volume
    fig.add_trace(go.Scatter(
        x=data.index, y=data[col_name],
        mode='lines', name=f'Average Volume {period}',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title=f'{ticker} - Volume and {period}-Day Average',
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
        raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

    fig = go.Figure()

    # Momentum
    fig.add_trace(go.Scatter(
        x=data.index, y=data[col_name],
        mode='lines', name=f'Momentum {period}',
        line=dict(color='purple')
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color='black', dash='dash'), annotation_text='Zero',
                  annotation_position="top left")

    fig.update_layout(
        title=f'{ticker} - Momentum {period} Days',
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

    # Original series plot
    fig.add_trace(go.Scatter(
        x=data.index, y=data[column], mode='lines', name='Original Series',
        line=dict(color='blue')
    ))

    # Differenced series plot
    if diff_column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[diff_column], mode='lines', name='Differenced Series',
            line=dict(color='orange')
        ))

    fig.update_layout(
        title="Stationarity: Original and Differenced Series",
        xaxis_title="",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=0.5)
    )

    return fig


# ARIMA graph
def plot_arima_predictions(data, arima_model, column='close'):
    predictions = arima_model.predict(start=data.index[0], end=data.index[-1])

    fig = go.Figure()

    # Actual series
    fig.add_trace(go.Scatter(
        x=data.index, y=data[column], mode='lines', name='Actual',
        line=dict(color='blue')
    ))

    # ARIMA predictions
    fig.add_trace(go.Scatter(
        x=data.index, y=predictions, mode='lines', name='ARIMA Predictions',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="Comparison of Actual Series and ARIMA Predictions",
        xaxis_title="",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=0.5
    ))

    return fig
# GARCH graph
def plot_garch_volatility(data, garch_model):
    conditional_volatility = garch_model.conditional_volatility

    fig = go.Figure()

    # Volatility series
    fig.add_trace(go.Scatter(
        x=data.index, y=conditional_volatility, mode='lines', name='Conditional Volatility',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title="Estimated Conditional Volatility (GARCH)",
        xaxis_title="",
        yaxis_title="Volatility",
        template="plotly_white",
    )

    return fig

def plot_var_analysis(data, var_value, var_h, cvar_95, confidence_level=0.95):

    if 'daily_returns' not in data.columns:
        print("Error: The 'daily_returns' column is not present in the data.")
        return

    daily_returns = data['daily_returns'].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_returns,
        nbinsx=50,
        name='Daily Returns',
        opacity=0.75,
        marker_color='#1f77b4'
    ))

    # VaR line
    fig.add_vline(x=var_value, line=dict(color='red', dash='dash'),
                  annotation_text=f"VaR ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    # Historical VaR line
    fig.add_vline(x=var_h, line=dict(color='orange', dash='dot'),
                  annotation_text=f"Historical VaR ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    # CVaR line
    fig.add_vline(x=cvar_95, line=dict(color='green', dash='solid'),
                  annotation_text=f"CVaR ({confidence_level * 100:.0f}%)",
                  annotation_position="top left")

    fig.update_layout(
        title="VaR, Historical VaR, and CVaR Analysis",
        xaxis_title="Daily Returns",
        yaxis_title="Frequency",
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
    Plots on the same graph:
      - The historical series (train + test)
      - The real values from the test set (in green)
      - The predictions on the test set for each model
      - The future predictions of each model

    Parameters:
    ----------
    data : pd.DataFrame
        Must contain at least the 'close' column. The index is of datetime type.
    results : dict
        Dictionary with model keys, e.g., "LSTM", "CNN+LSTM", "Transformer".
        Each key should map to a dictionary with:
          {
            'y_test_rescaled': np.array,
            'test_pred_rescaled': np.array,
            'forecast_df': pd.DataFrame with a 'forecast' column,
            ...
          }
    split_idx : int
        Index that separates the end of the train from the start of the test in 'data'.
        Used to correctly align the test set on the graph.
    """

    if data is None or len(data) == 0:
        raise ValueError("DataFrame 'data' is empty or None.")
    if 'close' not in data.columns:
        raise ValueError("The historical DataFrame must contain the 'close' column.")
    if not isinstance(results, dict) or len(results) == 0:
        raise ValueError("The 'results' parameter must be a non-empty dictionary.")
    if split_idx is None or not isinstance(split_idx, int):
        raise ValueError("split_idx must be a valid integer.")
    if split_idx < 0 or split_idx >= len(data):
        raise ValueError("split_idx is out of the DataFrame index bounds.")


    fig = go.Figure()

    # 1) Historical series (train + test) in blue
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))

    # 2) Real part (only test set) in green
    #    The test set starts from split_idx + 1 and lasts len(...) samples,
    #    but the models in results should have the same test set length.
    #    We get the length from the first model found, just to be safe.
    first_model_key = next(iter(results))  # e.g. 'LSTM'
    y_test_rescaled = results[first_model_key]['y_test_rescaled']
    n_test = len(y_test_rescaled)

    test_index = data.index[split_idx + 1: split_idx + 1 + n_test]

    fig.add_trace(go.Scatter(
        x=test_index,
        y=y_test_rescaled.flatten(),
        mode='lines',
        name='Actual (test set)',
        line=dict(color='green')
    ))

    model_colors = {
        "LSTM": "red",
        "CNN+LSTM": "orange",
        "Transformer": "purple"
    }
    # For future predictions, we use dashed lines
    dash_styles = {
        "LSTM": "solid",
        "CNN+LSTM": "dot",
        "Transformer": "dash"
    }

    # Add the test predictions and future forecasts for each model
    for model_name, model_dict in results.items():
        # (a) Predictions on test set
        test_pred_rescaled = model_dict['test_pred_rescaled']
        if test_pred_rescaled is None:
            continue
        if len(test_pred_rescaled) != n_test:
            raise ValueError(f"test_pred_rescaled size for {model_name} does not match y_test_rescaled.")

        fig.add_trace(go.Scatter(
            x=test_index,
            y=test_pred_rescaled.flatten(),
            mode='lines+markers',
            name=f'Test Pred. ({model_name})',
            line=dict(color=model_colors.get(model_name, 'gray')),
            showlegend=True
        ))

        # (b) Future predictions
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
        title="Multi-Model Forecast Comparison",
        xaxis_title="",
        yaxis_title="Price",
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
        print(f"Error: The 'close' column is not present in the data for the ticker '{ticker}'.")
        return

    decomposition = sm.tsa.seasonal_decompose(data['close'].dropna(), model='additive')

    fig = go.Figure()

    # Seasonality
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonality',
        line=dict(color='#2ca02c')
    ))

    # Residual
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residual',
        line=dict(color='#d62728')
    ))

    fig.update_layout(
        title=f"Seasonality and Residual for {ticker}",
        xaxis_title="",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    return fig

