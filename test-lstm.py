# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

from src.Config.Config import Config
from src.Data.DataSampler import DataSampler

if __name__ == '__main__':
    y_train = np.array([1, 2, 3, 4, 5])
    y_test = np.array([6, 7, 8, 9, 10])
    print(y_train, y_test)
    x = np.concatenate([y_train, y_test])
    print(x)
    dd
    # Fetch Apple stock data
    ticker = "AAPL"
    data = yf.download(ticker, start="2010-01-01", end="2022-01-01", progress=False)
    print('data : ', data)
    # Extract the closing prices
    df = data['Close'].values
    print('close : ', df)
    df = data['Close'].values.reshape(-1, 1)
    print('close reshaped : ', df)
    dd
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    print('df fit: ', df)
    X, y = DataSampler.split_sequences(Config.multivariate, df)
    print('X shape: ', X)
    print('X : ', X)
    print('y.shape : ', y.shape)
    print('y : ', y)
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('X_train shape: ', X_train)
    print('X_train : ', X_train)
    print('X_test shape : ', X_test.shape)
    print('X_test : ', X_test)
    print('y_train shape : ', y_train.shape)
    print('y_train : ', y_train)
    print('y_test shape : ', y_test.shape)
    print('y_test : ', y_test)
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Evaluate the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Transform predictions back to original scale
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test)

    # Calculate errors
    train_rmse = np.sqrt(np.mean(np.square(train_predictions - y_train)))
    test_rmse = np.sqrt(np.mean(np.square(test_predictions - y_test)))

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Combine Training and Test Predictions with their respective actual prices
    combined_dates_train = data.index[:len(y_train)]
    combined_actual_prices_train = y_train
    combined_predictions_train = train_predictions

    combined_dates_test = data.index[len(y_train):len(y_train) + len(y_test)]
    combined_actual_prices_test = y_test
    combined_predictions_test = test_predictions

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot Training Set
    plt.plot(combined_dates_train, combined_actual_prices_train, label="Actual Train Prices", color='blue')
    plt.plot(combined_dates_train, combined_predictions_train, label="Predicted Train Prices", color='orange',
             linestyle='dashed')

    # Plot Test Set
    plt.plot(combined_dates_test, combined_actual_prices_test, label="Actual Test Prices", color='green')
    plt.plot(combined_dates_test, combined_predictions_test, label="Predicted Test Prices", color='red',
             linestyle='dashed')

    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

from sklearn.preprocessing import MinMaxScaler

def run_for_univariate_series_ir_spiltted(dataset, models):
    dataset = dataset[Config.prediction_col].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    results = {}

    for model_name, model_config in models.items():
        model_result = Univariate.splitted_univariate_series(model_config, dataset, scaler)
        results[model_name] = model_result

    return results

# Example usage:
models_to_run = {
    'U-' + Config.CNN: Config.CNN,
    'U-' + Config.LSTM: Config.LSTM,
    'U-' + Config.bi_LSTM: Config.bi_LSTM,
    'U-' + Config.GRU: Config.GRU,
    'U-' + Config.bi_GRU: Config.bi_GRU,
    'U-' + Config.ANN: Config.ANN,
    'U-' + Config.bi_ANN: Config.bi_ANN,
    'U-' + Config.RNN: Config.RNN,
    'U-' + Config.bi_RNN: Config.bi_RNN,
}

results = run_for_univariate_series_ir_spiltted(your_dataset, models=models_to_run)

# Access individual model results
print(results['U-' + Config.CNN])
print(results['U-' + Config.LSTM])
# ... and so on
