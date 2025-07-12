from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
scaler = MinMaxScaler(feature_range=(0, 1))

# ===================== Data Loading & Preprocessing =====================

def get_stock_data():
    file_path = 'FinalDataset1.csv'
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values(by='Date').set_index('Date')
    if 'Stock' in df.columns:
        df['Stock'] = df['Stock'].astype(str).str.strip().str.upper()
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace({',': ''}, regex=True).astype(float)
    return df

df = get_stock_data()

def prepare_data(dataframe, time_step=60):
    scaler_local = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler_local.fit_transform(dataframe[['Close']])
    X, y = [], []
    for i in range(len(df_scaled) - time_step - 1):
        X.append(df_scaled[i:i+time_step, 0])
        y.append(df_scaled[i+time_step, 0])
    return np.array(X), np.array(y), scaler_local

# ===================== Model Definitions =====================

def build_lstm():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_rnn():
    model = Sequential([
        SimpleRNN(units=64, activation='relu', return_sequences=False, input_shape=(60, 1)),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ===================== Plotting Functions =====================

def plot_predictions(model, X_test, y_test, scaler_local, model_name):
    y_pred = model.predict(X_test)
    y_pred_inv = scaler_local.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_local.inverse_transform(y_test.reshape(-1, 1))
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Actual Price', color='blue')
    plt.plot(y_pred_inv, label='Predicted Price', color='red', linestyle='dashed')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(f'static/{model_name}_predictions.png', dpi=300)
    plt.close()

# ===================== Routes =====================

@app.route('/')
def home():
    stocks = df['Stock'].unique() if 'Stock' in df.columns else ['DEFAULT']
    return render_template('index.html', stocks=stocks)

@app.route('/train', methods=['GET'])
def train_models():
    stock_df = df[df['Stock'] == df['Stock'].unique()[0]] if 'Stock' in df.columns else df
    stock_df = stock_df[['Close']].dropna()
    
    time_step = 60
    X, y, scaler_local = prepare_data(stock_df, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    lstm_model = build_lstm()
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
    lstm_model.save('lstm_model.h5')
    lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)
    plot_predictions(lstm_model, X_test, y_test, scaler_local, "LSTM")
    
    rnn_model = build_rnn()
    rnn_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
    rnn_model.save('rnn_model.h5')
    rnn_loss = rnn_model.evaluate(X_test, y_test, verbose=0)
    plot_predictions(rnn_model, X_test, y_test, scaler_local, "RNN")
    
    return jsonify({
        "LSTM Test Loss": float(lstm_loss),
        "RNN Test Loss": float(rnn_loss),
        "LSTM Plot": "/static/LSTM_predictions.png",
        "RNN Plot": "/static/RNN_predictions.png"
    })

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name'].strip().upper()
    stock_data = df[df['Stock'] == stock_name] if 'Stock' in df.columns else df

    if stock_data.empty:
        return f"Stock '{stock_name}' not found in dataset!"

    scaled_data = scaler.fit_transform(stock_data[['Close']])
    if len(scaled_data) < 60:
        return "Not enough data to make predictions."

    X_seq = np.array([scaled_data[i-60:i] for i in range(60, len(scaled_data))])

    lstm_model = load_model("lstm_model.h5")
    rnn_model = load_model("rnn_model.h5")

    y_pred_lstm = lstm_model.predict(X_seq, batch_size=32).flatten()
    y_pred_rnn = rnn_model.predict(X_seq, batch_size=32).flatten()

    y_pred_lstm = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
    y_pred_rnn = scaler.inverse_transform(y_pred_rnn.reshape(-1, 1)).flatten()

    actual_prices = stock_data['Close'].values[60:len(y_pred_lstm)+60]

    def calculate_metrics(actual, predicted):
        return {
            "MAE": mean_absolute_error(actual, predicted),
            "MSE": mean_squared_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted))
        }

    lstm_metrics = calculate_metrics(actual_prices, y_pred_lstm)
    rnn_metrics = calculate_metrics(actual_prices, y_pred_rnn)

    errors = {"LSTM": lstm_metrics["MAE"], "RNN": rnn_metrics["MAE"]}
    best_model = min(errors, key=errors.get)

    def create_graph(predictions, title, color, metrics):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data.index[60:len(predictions)+60],
            open=stock_data['Open'][60:len(predictions)+60],
            high=stock_data['High'][60:len(predictions)+60],
            low=stock_data['Low'][60:len(predictions)+60],
            close=stock_data['Close'][60:len(predictions)+60],
            name='Actual Prices'
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index[60:len(predictions)+60],
            y=predictions,
            mode='lines',
            name=f"{title} (MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f})",
            line=dict(color=color)
        ))
        fig.update_layout(title=f"{title} Predictions", xaxis_title='Date', yaxis_title='Price',
                          width=1200, height=800)
        fig.write_html(f"static/{title}_graph.html")

    create_graph(y_pred_lstm, "LSTM", "red", lstm_metrics)
    create_graph(y_pred_rnn, "RNN", "green", rnn_metrics)

    return render_template('result.html', stock_name=stock_name, best_model=best_model,
                           lstm_graph="static/LSTM_graph.html",
                           rnn_graph="static/RNN_graph.html")

@app.route('/future_predict', methods=['POST'])
def future_predict():
    stock_name = request.form['stock_name'].strip().upper()
    n_days = int(request.form['n_days'])
    stock_data = df[df['Stock'] == stock_name] if 'Stock' in df.columns else df

    if stock_data.empty:
        return f"Stock '{stock_name}' not found!"

    scaled_data = scaler.fit_transform(stock_data[['Close']])
    X_input = scaled_data[-60:].reshape(1, 60, 1)

    lstm_model = load_model("lstm_model.h5")
    future_predictions = []
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=n_days)

    for _ in range(n_days):
        pred = lstm_model.predict(X_input)[0][0]
        future_predictions.append(pred)
        X_input = np.append(X_input[:, 1:, :], [[[pred]]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(),
                             mode='lines+markers', name='Future Predictions'))
    fig.update_layout(title="Future Stock Price Prediction",
                      xaxis_title="Date", yaxis_title="Predicted Price")
    fig.write_html("static/Future_graph.html")

    return render_template('future_result.html', future_graph="static/Future_graph.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

if __name__ == '__main__':
    app.run(debug=True)
