import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import logging
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras import layers


###############################################################################
# 1) Funzioni di utilità
###############################################################################

def slow_print(text, delay=0.01):
    print(text)


def create_sequences_multistep(X, y, window=10, horizon=30):
    """
    Trasforma i dati in sequenze per un modello multi-step diretto.
    Ritorna:
      Xs: (M, window, n_features)
      ys: (M, horizon)
    """
    Xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i:i + window])
        ys.append(y[i + window:i + window + horizon])
    return np.array(Xs), np.array(ys)


###############################################################################
# 2) MODELLI: LSTM, CNN+LSTM, TRANSFORMER
###############################################################################

def build_lstm_model(input_shape, lstm_units=128, horizon=30):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=lstm_units,
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=lstm_units // 2, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(horizon, activation='linear'))
    return model


def build_cnn_lstm_model(input_shape, lstm_units=128, horizon=30):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, padding='causal',
                            activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(lstm_units // 2, return_sequences=False))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(horizon, activation='linear'))
    return model


def build_transformer_model(input_shape, d_model=64, num_heads=4, ff_dim=256, num_layers=2, horizon=30):
    inputs = layers.Input(shape=input_shape)  # (window, n_features)

    # Proiezione d_model
    x = layers.Dense(d_model)(inputs)

    # Positional Embedding trainabile
    x = PositionalEmbedding(window_size=input_shape[0], d_model=d_model)(x)

    # Stack di blocchi Transformer
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim)(x)

    # GlobalAveragePooling1D
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(horizon, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


###############################################################################
# Strati di supporto per il Transformer
###############################################################################

class PositionalEmbedding(layers.Layer):
    def __init__(self, window_size, d_model):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.pos_embed = layers.Embedding(input_dim=window_size, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.window_size, delta=1)
        embedded_positions = self.pos_embed(positions)
        return x + embedded_positions


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def calculate_split_idx(data, test_size=0.2):
    """
    Calcola l'indice che separa la parte di train
    dalla parte di test, sulla base di 'test_size'.
    """
    if data is None:
        raise ValueError("I dati non possono essere None.")
    split_idx = int(len(data) * (1 - test_size))
    return split_idx


###############################################################################
# 3) Funzione Principale: addestra TUTTI I MODELLI (LSTM, CNN+LSTM, Transformer)
###############################################################################

def run_ai_price_forecast(
        data,
        arima_model,
        garch_result,
        benchmark_data=None,
        future_steps=30
):
    BOLD = "\033[1m"
    END = "\033[0m"
    print("")
    print(f"{BOLD}{'***************************************************************'}{END}")
    print(f"{BOLD}{'    PREVISIONE PREZZO CON AI (Multi-Step) - Tutti i modelli'}{END}")
    print(f"{BOLD}{'***************************************************************'}{END}")

    ##################################################################
    # 1) Chiede SOLO il livello di accuratezza/potenza di calcolo
    ##################################################################
    print("\nSeleziona il livello di accuratezza/potenza di calcolo:")
    print("  1) Leggera  -> meno epoche, rete piccola")
    print("  2) Media    -> compromesso intermedio")
    print("  3) Max      -> massime epoche, rete più grande\n")

    choice = input(
        f"{BOLD}{'  >>>'}{END} Inserisci {BOLD}{'1'}{END},  {BOLD}{'2'}{END}  o  {BOLD}{'3'}{END} :'  "
    ).strip()

    if choice == '1':
        EPOCHS = 50
        BATCH_SIZE = 32
        LSTM_UNITS = 64
        print("\nHai scelto la modalità Leggera.")
    elif choice == '3':
        EPOCHS = 500
        BATCH_SIZE = 300
        LSTM_UNITS = 300
        print("\nHai scelto la modalità Max.")
    else:
        EPOCHS = 100
        BATCH_SIZE = 128
        LSTM_UNITS = 128
        print("\nHai scelto la modalità Media (default).")

    ##################################################################
    # 2) Preparazione Dati
    ##################################################################
    try:
        data['garch_volatility'] = garch_result.conditional_volatility
    except:
        data['garch_volatility'] = np.nan

    try:
        arima_pred = arima_model.predict(start=data.index[0], end=data.index[-1])
        data['arima_pred'] = arima_pred
    except:
        data['arima_pred'] = np.nan

    if benchmark_data is not None:
        benchmark_data = benchmark_data.reindex(data.index)
        data['bench_ret'] = benchmark_data.get('daily_returns', 0.0)
    else:
        data['bench_ret'] = 0.0

    data['rolling_std_5'] = data['daily_returns'].rolling(5).std()
    data['day_of_week'] = data.index.dayofweek

    data.interpolate(method='linear', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.dropna(how='all', inplace=True)

    feature_cols = [
        'close', 'daily_returns', 'garch_volatility',
        'arima_pred', 'bench_ret', 'rolling_std_5', 'day_of_week'
    ]

    X_data = data[feature_cols].values
    y_data = data['close'].values

    # ================================================================
    # Gestione y come 2D per scalare correttamente
    # ================================================================
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)

    scaler_y = MinMaxScaler()
    y_data_2D = y_data.reshape(-1, 1)  # (N,1)
    y_scaled_2D = scaler_y.fit_transform(y_data_2D)  # (N,1)
    y_scaled = y_scaled_2D.flatten()

    TIME_STEPS = 10
    X_seq, y_seq = create_sequences_multistep(
        X_scaled, y_scaled,
        window=TIME_STEPS,
        horizon=future_steps
    )

    split_index = int(len(X_seq) * 0.8)
    X_train_full = X_seq[:split_index]
    y_train_full = y_seq[:split_index]
    X_test = X_seq[split_index:]
    y_test = y_seq[split_index:]

    ##################################################################
    # 3) Definisce tutti i modelli da allenare
    ##################################################################
    models_to_train = {
        "LSTM": lambda: build_lstm_model(
            input_shape=(TIME_STEPS, X_train_full.shape[2]),
            lstm_units=LSTM_UNITS,
            horizon=future_steps
        ),
        "CNN+LSTM": lambda: build_cnn_lstm_model(
            input_shape=(TIME_STEPS, X_train_full.shape[2]),
            lstm_units=LSTM_UNITS,
            horizon=future_steps
        ),
        "Transformer": lambda: build_transformer_model(
            input_shape=(TIME_STEPS, X_train_full.shape[2]),
            d_model=128,
            num_heads=8,
            ff_dim=256,
            num_layers=3,
            horizon=future_steps
        )
    }

    ##################################################################
    # 4) Per ciascun modello: TSCV, trova best model, valuta test
    ##################################################################
    results = {}

    # oggetto TimeSeriesSplit (Cross Validation)
    tscv = TimeSeriesSplit(n_splits=3)

    for model_name, build_fn in models_to_train.items():
        print(f"\n======================== {model_name} ========================")

        best_val_loss = np.inf
        best_weights = None  # Per salvare i pesi migliori
        tf.keras.optimizers.Adam(learning_rate=0.001)

        fold_idx = 0
        for train_idx, val_idx in tscv.split(X_train_full):
            fold_idx += 1
            print(f"\n=== TimeSeriesSplit Fold {fold_idx} ({model_name}) ===")

            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            # (Ri)costruisci il modello
            model = build_fn()


            # mette un LR fisso + ReduceLROnPlateau
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
            checkpoint_path = f"best_{model_name}_fold{fold_idx}.h5"
            checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                         monitor='val_loss', save_best_only=True, verbose=1)
            log_dir = f"logs/fit/{model_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-fold{fold_idx}"
            tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)

            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                shuffle=False,
                callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_cb],
                verbose=1
            )

            # calcola pesi migliori da checkpoint:
            model.load_weights(checkpoint_path)

            current_val_loss = min(history.history['val_loss'])
            print(f"Fold {fold_idx} val_loss minimo: {current_val_loss:.6f}")

            # Se è il migliore di tutti i fold, aggiorna best_weights
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights = model.get_weights()

        # Fine TSCV, ricrea un modello "pulito" e carica i best_weights
        if best_weights is None:
            print(f"ERRORE: nessun modello {model_name} addestrato correttamente!")
            continue

        best_model = build_fn()

        # Compila il modello
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )

        best_model.set_weights(best_weights)  # Carichi i "best_weights"
        test_loss = best_model.evaluate(X_test, y_test, verbose=0)

        print(f"\n=== RISULTATI FINALI {model_name} ===")
        print(f"Val Loss migliore (CV): {best_val_loss:.6f}")
        print(f"Test Loss: {test_loss:.6f}")

        # Predizioni di test (M, horizon)
        test_pred = best_model.predict(X_test)

        # ================================================================
        # inverse_transform su 2D coerente
        # ================================================================
        # test_pred.shape = (M, horizon)
        # y_test.shape    = (M, horizon)

        # Reshape per adattare il MinMaxScaler
        test_pred_2D = test_pred.reshape(-1, 1)  # (M*horizon, 1)
        test_pred_rescaled_2D = scaler_y.inverse_transform(test_pred_2D)
        test_pred_rescaled = test_pred_rescaled_2D.reshape(-1, future_steps)

        y_test_2D = y_test.reshape(-1, 1)  # (M*horizon, 1)
        y_test_rescaled_2D = scaler_y.inverse_transform(y_test_2D)
        y_test_rescaled = y_test_rescaled_2D.reshape(-1, future_steps)

        # Metriche
        mae_all = []
        mape_all = []
        for step in range(future_steps):
            mae_step = mean_absolute_error(y_test_rescaled[:, step], test_pred_rescaled[:, step])
            mape_step = mean_absolute_percentage_error(y_test_rescaled[:, step], test_pred_rescaled[:, step])
            mae_all.append(mae_step)
            mape_all.append(mape_step)

        mean_mae = np.mean(mae_all)
        mean_mape = np.mean(mape_all) * 100
        print(f"MAE medio su {future_steps} passi futuri: {mean_mae:.3f}")
        print(f"MAPE medio su {future_steps} passi futuri: {mean_mape:.2f}%")

        # Previsione FUTURA (direct multi-step)
        last_input = X_test[-1:]  # shape (1, TIME_STEPS, n_features)
        future_prediction = best_model.predict(last_input)[0]  # shape (future_steps,)

        future_pred_2D = future_prediction.reshape(-1, 1)  # (future_steps, 1)
        future_pred_rescaled_2D = scaler_y.inverse_transform(future_pred_2D)
        future_pred_rescaled = future_pred_rescaled_2D.flatten()  # shape (future_steps,)

        forecast_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='D')[1:]
        forecast_df = pd.DataFrame({'forecast': future_pred_rescaled}, index=forecast_index)

        print("\n[ PREVISIONE FUTURA DIRETTA ]")
        print(forecast_df)

        # Salva i risultati in un dizionario
        results[model_name] = {
            'best_model': best_model,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'forecast_df': forecast_df,
            'y_test_rescaled': y_test_rescaled,
            'test_pred_rescaled': test_pred_rescaled,
            'mean_mae': mean_mae,
            'mean_mape': mean_mape
        }

    print("")
    print(f"{BOLD}{'***************************************************************'}{END}")
    print(f"{BOLD}{'Fine Funzione AI Price Forecast (Multi-Step) - Tutti i modelli'}{END}")
    print(f"{BOLD}{'***************************************************************'}{END}")
    print("")

    return results


######## tabella risultati

def build_forecast_table(ai_results):
    """
    Crea un unico DataFrame che contiene, in ogni colonna,
    le previsioni di un modello diverso e un'unica colonna di date.

    Ritorna: (merged_df, html_table)
      merged_df: il DataFrame unificato (indice = date, colonne = nome modello)
      html_table: stringa HTML con la tabella
    """
    if not ai_results or not isinstance(ai_results, dict):
        return None, "<p>Nessuna previsione disponibile.</p>"

    merged_df = pd.DataFrame()

    for model_name, model_info in ai_results.items():
        fdf = model_info.get('forecast_df')
        if fdf is None or 'forecast' not in fdf.columns:
            continue

        # Copia il DF per evitare avvisi SettingWithCopy
        temp_df = fdf.copy()
        # Rinomina la colonna 'forecast' con il nome del modello
        temp_df.rename(columns={'forecast': model_name}, inplace=True)

        # Se merged_df è vuoto, inizializza con questo DF
        if merged_df.empty:
            merged_df = temp_df
        else:
            # Outer join sull'indice (le date)
            merged_df = merged_df.join(temp_df, how='outer')

    if merged_df.empty:
        return None, "<p>Nessuna previsione valida trovata.</p>"

    # Imposta il nome dell'indice
    merged_df.index.name = "Data"


    #   reset_index() per portare 'Data' come colonna e non come indice
    html_table = merged_df.reset_index().to_html(index=False)

    return merged_df, html_table
