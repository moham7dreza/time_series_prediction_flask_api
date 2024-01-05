# # Import necessary libraries
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
#
# from src.Config.Config import Config
# from src.Data.DataSampler import DataSampler
#
# if __name__ == '__main__':
#     y_train = np.array([1, 2, 3, 4, 5])
#     y_test = np.array([6, 7, 8, 9, 10])
#     print(y_train, y_test)
#     x = np.concatenate([y_train, y_test])
#     print(x)
#     dd
#     # Fetch Apple stock data
#     ticker = "AAPL"
#     data = yf.download(ticker, start="2010-01-01", end="2022-01-01", progress=False)
#     print('data : ', data)
#     # Extract the closing prices
#     df = data['Close'].values
#     print('close : ', df)
#     df = data['Close'].values.reshape(-1, 1)
#     print('close reshaped : ', df)
#     dd
#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     df = scaler.fit_transform(df)
#     print('df fit: ', df)
#     X, y = DataSampler.split_sequences(Config.multivariate, df)
#     print('X shape: ', X)
#     print('X : ', X)
#     print('y.shape : ', y.shape)
#     print('y : ', y)
#     # Train-test split (80% train, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print('X_train shape: ', X_train)
#     print('X_train : ', X_train)
#     print('X_test shape : ', X_test.shape)
#     print('X_test : ', X_test)
#     print('y_train shape : ', y_train.shape)
#     print('y_train : ', y_train)
#     print('y_test shape : ', y_test.shape)
#     print('y_test : ', y_test)
#     # Build LSTM model
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
#     model.add(LSTM(units=50))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#
#     # Train the model
#     model.fit(X_train, y_train, epochs=50, batch_size=32)
#
#     # Evaluate the model
#     train_predictions = model.predict(X_train)
#     test_predictions = model.predict(X_test)
#
#     # Transform predictions back to original scale
#     train_predictions = scaler.inverse_transform(train_predictions)
#     y_train = scaler.inverse_transform(y_train)
#     test_predictions = scaler.inverse_transform(test_predictions)
#     y_test = scaler.inverse_transform(y_test)
#
#     # Calculate errors
#     train_rmse = np.sqrt(np.mean(np.square(train_predictions - y_train)))
#     test_rmse = np.sqrt(np.mean(np.square(test_predictions - y_test)))
#
#     print(f"Train RMSE: {train_rmse}")
#     print(f"Test RMSE: {test_rmse}")
#
#     # Combine Training and Test Predictions with their respective actual prices
#     combined_dates_train = data.index[:len(y_train)]
#     combined_actual_prices_train = y_train
#     combined_predictions_train = train_predictions
#
#     combined_dates_test = data.index[len(y_train):len(y_train) + len(y_test)]
#     combined_actual_prices_test = y_test
#     combined_predictions_test = test_predictions
#
#     # Plot results
#     plt.figure(figsize=(12, 6))
#
#     # Plot Training Set
#     plt.plot(combined_dates_train, combined_actual_prices_train, label="Actual Train Prices", color='blue')
#     plt.plot(combined_dates_train, combined_predictions_train, label="Predicted Train Prices", color='orange',
#              linestyle='dashed')
#
#     # Plot Test Set
#     plt.plot(combined_dates_test, combined_actual_prices_test, label="Actual Test Prices", color='green')
#     plt.plot(combined_dates_test, combined_predictions_test, label="Predicted Test Prices", color='red',
#              linestyle='dashed')
#
#     plt.title("Actual vs Predicted Prices")
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# from sklearn.preprocessing import MinMaxScaler
#
# def run_for_univariate_series_ir_spiltted(dataset, models):
#     dataset = dataset[Config.prediction_col].values.reshape(-1, 1)
#
#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     dataset = scaler.fit_transform(dataset)
#
#     results = {}
#
#     for model_name, model_config in models.items():
#         model_result = Univariate.splitted_univariate_series(model_config, dataset, scaler)
#         results[model_name] = model_result
#
#     return results
#
# # Example usage:
# models_to_run = {
#     'U-' + Config.CNN: Config.CNN,
#     'U-' + Config.LSTM: Config.LSTM,
#     'U-' + Config.bi_LSTM: Config.bi_LSTM,
#     'U-' + Config.GRU: Config.GRU,
#     'U-' + Config.bi_GRU: Config.bi_GRU,
#     'U-' + Config.ANN: Config.ANN,
#     'U-' + Config.bi_ANN: Config.bi_ANN,
#     'U-' + Config.RNN: Config.RNN,
#     'U-' + Config.bi_RNN: Config.bi_RNN,
# }
#
# results = run_for_univariate_series_ir_spiltted(your_dataset, models=models_to_run)
#
# # Access individual model results
# print(results['U-' + Config.CNN])
# print(results['U-' + Config.LSTM])
# # ... and so on
#
#
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
#
# class MultivariateTimeSeriesModel:
#     @staticmethod
#     def split_sequences(config, dataset):
#         # Your code to split sequences goes here
#         pass
#
# class MultivariateModelBuilder:
#     @staticmethod
#     def get_model(model_name, n_features):
#         # Your code to build the model goes here
#         pass
#
# class Config:
#     univariate = 10
#     test_size = 0.2
#     random_state = 42
#     epochs_for_multivariate_series = 10
#     checkForModelExistsInFolder = True
#     drive_model_folder_path = "/path/to/drive_model_folder"
#     local_model_folder_path = "/path/to/local_model_folder"
#     colab = False
#
# class Helper:
#     @staticmethod
#     def merge_and_clean(round_decimals, arr1, arr2):
#         # Your code to merge and clean arrays goes here
#         pass
#
# def splitted_multivariate_series(model_name, dataset, scaler, dates):
#     # Split into samples
#     X, y = MultivariateTimeSeriesModel.split_sequences(Config.univariate, dataset)
#
#     # Reshape from [samples, timesteps] into [samples, timesteps, features]
#     n_features = X.shape[2]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.test_size, random_state=Config.random_state)
#
#     # Define the path for saving/loading the model
#     if Config.colab:
#         model_path = os.path.join(Config.drive_model_folder_path, 'M-' + model_name + '.h5')
#     else:
#         model_path = os.path.join(Config.local_model_folder_path, 'M-' + model_name + '.h5')
#
#     # Check if the model file exists
#     if Config.checkForModelExistsInFolder and os.path.exists(model_path):
#         # Load the existing model
#         model = load_model(model_path)
#         print("Model '{}' loaded from file.".format('M-' + model_name))
#     else:
#         # Define model
#         model = MultivariateModelBuilder.get_model(model_name, n_features)
#         # Fit model
#         model.fit(X_train, y_train, epochs=Config.epochs_for_multivariate_series, batch_size=32)
#
#         # Save the model
#         model.save(model_path)
#         print("Model '{}' saved to file.".format('M-' + model_name))
#
#     loss = model.evaluate(X_test, y_test)
#     print("Model '{}' loss is: {}".format('M-' + model_name, loss))
#
#     # Evaluate the model
#     train_predictions = model.predict(X_train)
#     test_predictions = model.predict(X_test)
#
#     # Transform predictions back to original scale
#     train_predictions = scaler.inverse_transform(train_predictions)
#     y_train = scaler.inverse_transform(y_train)
#     test_predictions = scaler.inverse_transform(test_predictions)
#     y_test = scaler.inverse_transform(y_test)
#
#     # Calculate errors
#     train_rmse = np.sqrt(np.mean(np.square(train_predictions - y_train)))
#     test_rmse = np.sqrt(np.mean(np.square(test_predictions - y_test)))
#
#     print(f"Train RMSE: {train_rmse}")
#     print(f"Test RMSE: {test_rmse}")
#
#     actuals = Helper.merge_and_clean(round_decimals=2, arr1=y_train, arr2=y_test)
#     predictions = Helper.merge_and_clean(round_decimals=2, arr1=train_predictions, arr2=test_predictions)
#
#     # Check if all arrays have the same length
#     if len(dates) == len(actuals) == len(predictions):
#         # Create the mapping
#         data_mapping = {
#             index + 1: {"date": date, "actual": actual, "predict": predict}
#             for index, (date, actual, predict) in enumerate(zip(dates, actuals, predictions))
#         }
#         return data_mapping
#     else:
#         raise ValueError("Arrays must have the same length.")
#
# # Example usage:
# # Replace the following with your actual data, scaler, and dates
# example_dataset = np.random.rand(100, 5, 10)  # Replace with your actual dataset
# example_scaler = MinMaxScaler()  # Replace with your actual scaler
# example_dates = ['2023-01-01', '2023-01-02', '2023-01-03']  # Replace with your actual dates
#
# result_mapping = splitted_multivariate_series('your_model_name', example_dataset, example_scaler, example_dates)
# print(result_mapping)
#
#
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
#
# class MultivariateModelBuilder:
#     @staticmethod
#     def get_multivariate_cnn_model(input_shape, n_outputs):
#         visible = Input(shape=input_shape)
#         cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
#         cnn = MaxPooling1D(pool_size=2)(cnn)
#         cnn = Flatten()(cnn)
#         cnn = Dense(50, activation='relu')(cnn)
#
#         # Define multiple outputs
#         outputs = [Dense(1)(cnn) for _ in range(n_outputs)]
#
#         # Tie together
#         model = Model(inputs=visible, outputs=outputs)
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
# # Example usage:
# # Replace the following with your actual data, scaler, and dates
# example_dataset = np.random.rand(100, 5, 3)  # Replace with your actual dataset
# example_scaler = MinMaxScaler()  # Replace with your actual scaler
# example_dates = ['2023-01-01', '2023-01-02', '2023-01-03']  # Replace with your actual dates
#
# # Modify your data preparation code based on the provided multivariate sequence splitting function
# def split_multivariate_sequences(sequences, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequences)):
#         end_ix = i + n_steps
#         if end_ix > len(sequences) - 1:
#             break
#         seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)
#
# # Split your multivariate dataset
# n_steps = 3
# X, y = split_multivariate_sequences(example_dataset, n_steps)
#
# # Separate output
# n_outputs = example_dataset.shape[2]
# y_list = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_outputs)]
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train_list, y_test_list = train_test_split(X, y_list, test_size=0.2, random_state=42)
#
# # Build and train the model
# input_shape = (n_steps, example_dataset.shape[2])
# model = MultivariateModelBuilder.get_multivariate_cnn_model(input_shape, n_outputs)
# model.fit(X_train, y_train_list, epochs=2000, verbose=0)
#
# # Evaluate the model
# loss = model.evaluate(X_test, y_test_list)
# print("Model loss is:", loss)
#
# # Demonstrate prediction
# x_input = np.array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
# x_input = x_input.reshape((1, n_steps, example_dataset.shape[2]))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
#
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
#
# class MultivariateModelBuilder:
#     @staticmethod
#     def get_multivariate_cnn_model(input_shape, n_outputs):
#         visible = Input(shape=input_shape)
#         cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
#         cnn = MaxPooling1D(pool_size=2)(cnn)
#         cnn = Flatten()(cnn)
#         cnn = Dense(50, activation='relu')(cnn)
#
#         # Define multiple outputs
#         outputs = [Dense(1)(cnn) for _ in range(n_outputs)]
#
#         # Tie together
#         model = Model(inputs=visible, outputs=outputs)
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
# # Example usage:
# # Replace the following with your actual data, scaler, and dates
# example_dataset = np.random.rand(100, 5, 3)  # Replace with your actual dataset
# example_scaler = MinMaxScaler()  # Replace with your actual scaler
# example_dates = ['2023-01-01', '2023-01-02', '2023-01-03']  # Replace with your actual dates
#
# # Modify your data preparation code based on the provided multivariate sequence splitting function
# def split_multivariate_sequences(sequences, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequences)):
#         end_ix = i + n_steps
#         if end_ix > len(sequences) - 1:
#             break
#         seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)
#
# # Fit the scaler on the data and transform the dataset
# example_dataset_scaled = example_scaler.fit_transform(example_dataset.reshape(-1, example_dataset.shape[-1])).reshape(example_dataset.shape)
#
# # Split your multivariate dataset
# n_steps = 3
# X, y = split_multivariate_sequences(example_dataset_scaled, n_steps)
#
# # Separate output
# n_outputs = example_dataset.shape[2]
# y_list = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_outputs)]
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train_list, y_test_list = train_test_split(X, y_list, test_size=0.2, random_state=42)
#
# # Build and train the model
# input_shape = (n_steps, example_dataset.shape[2])
# model = MultivariateModelBuilder.get_multivariate_cnn_model(input_shape, n_outputs)
# model.fit(X_train, y_train_list, epochs=2000, verbose=0)
#
# # Evaluate the model
# loss = model.evaluate(X_test, y_test_list)
# print("Model loss is:", loss)
#
# # Demonstrate prediction using test data
# test_predictions_list = model.predict(X_test, verbose=0)
#
# # Inverse transform predictions and actuals back to original scale
# test_predictions = np.concatenate([example_scaler.inverse_transform(pred) for pred in test_predictions_list], axis=1)
# y_test = example_scaler.inverse_transform(np.concatenate(y_test_list, axis=1))
#
# print("Test Predictions:")
# print(test_predictions)
# print("Actuals:")
# print(y_test)
#
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.Config.Config import Config
from src.Helper.Helper import Helper
from src.Model.Evaluation import Evaluation

if __name__ == '__main__':
    print()
    # from sklearn.model_selection import train_test_split

    # Assume you have a dataset with features (X) and labels (y)
    # X should be a 2D array (or DataFrame), and y should be a 1D array (or Series)
    # Let's generate a sample dataset for demonstration purposes.

    # Import necessary libraries
    # import numpy as np

    # Generate a synthetic dataset
    # X = np.round(np.random.rand(10, 3, 5), 2)  # 100 samples, 3 features
    # X2 = np.random.rand(10, 3, 5)  # 100 samples, 3 features
    # actuals = [2376.6,2780.6,2889.25,2740.6,2596.0,2250.5,2054.25,2054.25,2054.25,2054.0,1862.75,1784.33,2005.0,2178.6,1995.67,1832.8,1866.2,1781.4,1693.0,1630.4,1789.5,1861.25,1958.6,1831.6,1777.75,1766.0,1958.8,2127.25,2060.8,1962.67,1874.2,1916.0,1997.2,2178.2,2081.8,2081.8,2021.0,2030.4,2112.4,2286.5,2483.25,2743.2,2617.0,2567.0,2705.2,2874.8,3213.0,3251.6,3171.25,2838.8,2873.67,2880.8,2686.0,2652.8,2634.2,2623.8,2693.4,2574.0,2567.0,2478.6,2678.2,2809.2,2953.5,3068.5,3180.5,3188.0,3083.6,3006.6,2698.5,2572.4,2409.5,2414.0,2389.0,2333.67,2271.2,2190.75,2280.0,2390.0,2542.0,2662.5,2418.75,2406.2,2377.0,2315.25,2244.8,2289.8,2467.8,2483.8,2352.6,2193.6,2085.0,1951.6,1862.8,1903.25,1861.5,1803.6,1796.4,2011.8,2040.8,1826.0,1826.0,1468.5,1392.2,1424.0,1629.6,1946.6,2460.0,3023.75,3507.8,3751.0,4565.6,4186.8,4829.6,5571.0,5571.0,5571.0,5922.67,5593.2,5239.4,4537.4,4422.2,4734.6,4444.8,3978.0,4383.75,4630.5,4548.6,4506.0,4239.67,4018.0,3848.5,3649.0,3414.5,3414.5,3414.5,3414.5,3414.5,3414.5,3414.5,3414.5,3414.5,3476.0,3743.6,3694.2,3684.0,3897.8,4072.0,3758.8,3677.4,3621.6,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,3443.33,2514.0,2411.0,2134.25,1982.0,1892.2,1968.0,2188.5,2254.5,2446.75,2455.8,2378.5,2271.0,2167.6,2185.2,2159.4,2062.2,1977.67,1983.4,1917.0,1953.33,1966.4,1976.2,1976.2,2001.25,1978.75,1968.4,1913.6,1871.8,1944.2,1922.5,1899.6,2015.8,2062.0,1906.0,1771.2,1789.8,1726.0,1706.6,1767.4,1715.4,1685.0,1619.0,1705.5,1817.2,1810.8,1910.8,1698.4,1765.2,1745.4,1737.6,1702.8,1806.0,1963.75,2216.75,2637.2,2578.6,2382.8,2521.5,2464.25,2402.67,2284.75,2554.2,2420.0,2365.5,2308.8,2733.0,3145.8,3266.25,3266.25,3144.0,3216.2,2943.6,3296.25,3288.75,3151.8,3073.8,3004.2,3000.4,3383.2,3493.25,3564.4,3860.8,4380.8,4901.5,5808.0,5458.8,4648.2,4321.4,4023.4,4110.5,3993.25,4081.6,4105.8,4316.0,4506.2,4460.2,4032.33,4412.6,4501.0,4501.0,4501.0,4501.0,4501.0,4501.0,4501.0,4501.0,1071.0,976.4,966.8,1099.67,1300.5,1535.0,1805.2,2105.6,1846.0,1818.4,1678.0,1602.6,1638.0,1799.0,2011.0,2424.4,2537.75,2221.0,2198.6,2263.2,2122.2,2236.8,2214.8,2508.6,2941.25,3120.5,3448.0,4111.4,4960.67,5494.75,5225.2,4745.6,4330.2,4010.5,3491.2,3566.33,3686.5,3774.8,3957.4,4300.2,4436.4,4658.2,4545.4,4612.6,4674.8,4023.0,4418.6,4837.2,5297.5,5532.0,5029.0,5502.4,6346.8,6981.0,6686.75,6591.75,7188.33,7920.33,8411.8,8443.0,8848.8,10539.2,12228.5,14128.2,13449.4,12639.33,11439.0,12084.6,12977.5,13668.0,13434.0,14500.0,16846.0,16118.0,13798.0,14037.5,12396.0,11256.0,11040.0,11002.5,10920.0,10520.0,10122.0,10104.0,9827.5,9335.0,9200.0,9197.5,9182.5,9135.0,7640.0,7064.0,7722.0,8066.0,8780.0,9056.0,8500.0,8494.0,8874.0,8570.0,8122.0,8416.0,8285.0,8066.0,8060.0,8060.0,8060.0,8020.0,8020.0,8020.0,8020.0,8020.0,8020.0,8020.0,8020.0,7886.0,7280.0,7230.0,7220.0,5595.0,5320.0,5708.0,6450.0,6350.0,6282.0,6995.0,6780.0,7238.0,7704.0,8230.0,8098.0,8062.0,9254.0,9144.0,9082.0,9150.0,9385.0,8386.0,7572.0,6510.0,6602.0,7528.0,7604.0,8012.5,7204.0,6594.0,6156.0,6374.0,7240.0,7388.0,6984.0,6466.0,5918.0,6034.0,5742.0,6197.5,5918.0,5897.5,5978.0,6060.0,6400.0,7030.0,8000.0,7886.0,7940.0,8198.0,8303.33,8716.0,9020.0,8037.5,7896.67,7640.0,7513.33,7520.0,7396.0,7372.0,7357.5,8040.0,7692.0,8034.0,8440.0,8876.0,9552.0,10180.0,10930.0,11782.5,11782.0,12160.0,11252.5]
    # m_ann = [1866.5899658203125,1943.8599853515625,1989.969970703125,1987.1400146484375,1979.3599853515625,1981.0,1944.4300537109375,1879.739990234375,1832.0899658203125,1786.18994140625,1780.1800537109375,1740.1300048828125,1663.47998046875,1644.300048828125,1665.6800537109375,1689.010009765625,1679.4000244140625,1622.1300048828125,1594.969970703125,1582.4100341796875,1575.699951171875,1523.8199462890625,1487.760009765625,1458.6500244140625,1513.5699462890625,1563.530029296875,1596.219970703125,1605.030029296875,1602.4100341796875,1604.5400390625,1598.25,1607.6400146484375,1598.9300537109375,1580.56005859375,1539.7099609375,1539.760009765625,1561.0899658203125,1604.449951171875,1645.550048828125,1703.56005859375,1731.8599853515625,1741.8699951171875,1753.4599609375,1775.760009765625,1823.9300537109375,1874.75,1916.5400390625,1842.56005859375,1786.4000244140625,1759.68994140625,1731.1800537109375,1679.719970703125,1651.010009765625,1517.31005859375,1604.93994140625,1504.780029296875,1506.1700439453125,1474.4300537109375,1466.5999755859375,1461.77001953125,1471.6700439453125,1507.239990234375,1542.6600341796875,1570.8699951171875,1580.510009765625,1573.530029296875,1523.1500244140625,1480.6700439453125,1439.0699462890625,1427.0,1394.6099853515625,1363.1700439453125,1303.6099853515625,1264.5699462890625,1259.489990234375,1272.0799560546875,1303.8499755859375,1323.6199951171875,1325.1300048828125,1311.300048828125,1290.4000244140625,1257.6800537109375,1239.949951171875,1238.489990234375,1259.969970703125,1275.8699951171875,1251.1700439453125,1197.8699951171875,1143.3599853515625,1102.260009765625,1073.5899658203125,1061.1400146484375,1042.3800048828125,1051.7099609375,1055.4000244140625,1088.1300048828125,1106.0699462890625,1097.1600341796875,1071.47998046875,1023.5,980.8800048828125,951.9400024414062,964.6699829101562,1022.6699829101562,1124.489990234375,1263.199951171875,1396.449951171875,1476.510009765625,1586.77001953125,1628.780029296875,1668.0899658203125,1745.02001953125,1818.77001953125,1855.969970703125,1898.3399658203125,1891.3399658203125,1848.4300537109375,1731.3599853515625,1640.8800048828125,1615.8199462890625,1612.9200439453125,1571.0400390625,1557.0999755859375,1579.489990234375,1599.47998046875,1587.699951171875,1549.8900146484375,1509.2900390625,1459.2900390625,1414.030029296875,1375.25,1349.1300048828125,1354.27001953125,1353.75,1350.93994140625,1347.25,1346.3399658203125,1342.739990234375,1344.0400390625,1347.5,1379.1400146484375,1396.2900390625,1412.06005859375,1431.27001953125,1467.6199951171875,1467.6099853515625,1441.1400146484375,1423.5699462890625,1400.52001953125,1398.4000244140625,1405.0400390625,1393.75,1382.75,1368.6700439453125,1338.0,1312.5,1324.3299560546875,1302.239990234375,1309.81005859375,1290.97998046875,1206.780029296875,1123.6700439453125,1036.3900146484375,996.8699951171875,958.8400268554688,953.8699951171875,976.739990234375,1021.3699951171875,1086.800048828125,1127.5400390625,1175.1300048828125,1173.010009765625,1142.8599853515625,1124.5799560546875,1110.52001953125,1091.550048828125,1075.760009765625,1044.72998046875,1025.050048828125,1018.5,1012.8800048828125,1029.739990234375,1034.9000244140625,1045.3199462890625,1048.510009765625,1049.989990234375,1027.06005859375,1013.5900268554688,1009.97998046875,1005.5499877929688,996.0,1000.1300048828125,1006.8900146484375,1010.75,1002.22998046875,979.1699829101562,964.719970703125,992.7899780273438,1014.739990234375,1017.9600219726562,1018.4000244140625,995.4600219726562,990.219970703125,1008.5,1011.1599731445312,1025.1600341796875,1024.300048828125,1027.780029296875,1004.6300048828125,1028.0400390625,1021.989990234375,1029.3499755859375,1056.1800537109375,1060.5400390625,1090.4300537109375,1166.7900390625,1201.0899658203125,1175.489990234375,1181.3299560546875,1192.3800048828125,1195.760009765625,1266.97998046875,1277.760009765625,1319.4599609375,1401.010009765625,1398.9200439453125,1412.300048828125,1482.489990234375,1523.81005859375,1577.3199462890625,1653.550048828125,1774.8499755859375,1818.5899658203125,1748.0,1776.1600341796875,1962.75,2250.2900390625,2177.239990234375,2102.9599609375,2191.1201171875,2251.389892578125,2633.0400390625,2869.31005859375,2842.659912109375,2911.9599609375,2971.280029296875,2966.47998046875,2911.679931640625,2836.60009765625,2799.340087890625,2778.139892578125,2523.7099609375,2511.070068359375,2573.68994140625,2452.0400390625,2328.47998046875,2177.409912109375,2277.25,2372.159912109375,2395.64990234375,2488.989990234375,2527.260009765625,2532.9599609375,2527.550048828125,2550.090087890625,2706.409912109375,2715.81005859375,2470.360107421875,2314.39990234375,2328.610107421875,2348.2900390625,2404.22998046875,2511.760009765625,2545.39990234375,2559.330078125,2637.2900390625,2754.550048828125,2759.52001953125,2573.219970703125,2455.050048828125,2415.56005859375,2392.75,2489.8798828125,2520.179931640625,2419.14990234375,2386.530029296875,2262.10009765625,2227.27001953125,2277.4599609375,2244.02001953125,2243.969970703125,2282.110107421875,2287.590087890625,2337.679931640625,2476.52001953125,2556.39990234375,2608.7900390625,2626.800048828125,2616.89990234375,2570.159912109375,2442.489990234375,2375.639892578125,2330.699951171875,2385.25,2469.889892578125,2561.93994140625,2639.090087890625,2817.0400390625,2888.510009765625,3017.639892578125,3180.830078125,3154.3701171875,3156.169921875,3030.52001953125,3319.889892578125,3518.489990234375,3549.110107421875,3709.3798828125,3929.260009765625,4407.22998046875,4566.22021484375,4596.5498046875,4600.72021484375,4903.52001953125,5355.16015625,5755.52001953125,5976.2900390625,6392.3701171875,7044.080078125,7680.85009765625,8020.41015625,8009.1298828125,7985.18994140625,8229.259765625,8679.8603515625,9316.3603515625,9592.33984375,9958.4697265625,10411.150390625,10307.73046875,9992.4599609375,9990.2197265625,10060.6103515625,10009.73046875,9934.7197265625,9417.9697265625,9129.9501953125,9389.6298828125,9542.669921875,9345.330078125,9037.6396484375,9068.7197265625,9030.2998046875,8540.509765625,8143.39013671875,7813.580078125,7550.02978515625,7268.31982421875,7417.6201171875,7756.52001953125,8063.58984375,8202.3603515625,8047.41015625,7887.419921875,7802.75,7497.2900390625,7483.39013671875,7717.419921875,7925.66015625,8478.740234375,8695.599609375,8769.33984375,8748.3203125,8738.919921875,8840.9599609375,8973.0703125,8878.0595703125,8764.0498046875,8641.990234375,8506.0,8244.16015625,8050.89013671875,8059.919921875,7816.77978515625,7584.7099609375,7259.93017578125,6931.35009765625,6895.39990234375,7035.5498046875,7149.39013671875,7160.56005859375,7140.240234375,7384.830078125,7553.33984375,7810.39013671875,8269.1201171875,8715.9404296875,8928.3203125,9006.83984375,9200.26953125,9099.4501953125,9223.8603515625,9235.26953125,8770.990234375,8267.099609375,7617.81982421875,7238.25,7187.0400390625,7133.4599609375,7206.6201171875,7089.06982421875,6864.5498046875,6586.22998046875,6614.6298828125,6657.43017578125,6802.0400390625,6791.68017578125,6791.64013671875,6555.72998046875,6652.97021484375,6358.60009765625,6594.0498046875,6622.68017578125,6586.5,6818.43994140625,7207.10986328125,7435.89013671875,7530.35986328125,7663.93994140625,7902.740234375,8175.4599609375,8389.900390625,8672.240234375,8812.900390625,8862.3603515625,8913.5595703125,8771.509765625,8638.16015625,8473.5595703125,8145.64013671875,7883.56005859375,7741.990234375,7679.080078125,7756.91015625,7766.259765625,7783.509765625,7887.0400390625,7941.10986328125,8161.240234375,8466.1298828125,8616.16015625,8635.490234375,8647.669921875,8500.9697265625]
    # m_b_ann = [807.1799926757812,824.5,838.739990234375,848.0999755859375,855.27001953125,871.72998046875,913.1799926757812,949.1699829101562,946.969970703125,936.4600219726562,938.02001953125,936.5999755859375,941.0499877929688,968.52001953125,988.77001953125,991.75,988.3099975585938,987.6699829101562,991.5399780273438,993.6599731445312,983.1799926757812,970.6699829101562,961.0499877929688,946.6799926757812,947.22998046875,947.6500244140625,952.2100219726562,957.0800170898438,953.1799926757812,945.719970703125,940.8800048828125,948.3599853515625,953.0900268554688,958.010009765625,961.8800048828125,967.1900024414062,982.1199951171875,985.97998046875,993.3599853515625,995.0900268554688,998.7000122070312,1000.1599731445312,1004.0499877929688,1005.9400024414062,1015.0,1028.6700439453125,1064.5899658203125,1069.030029296875,1085.5400390625,1081.8599853515625,1079.9300537109375,1061.0400390625,1058.4300537109375,1064.27001953125,1073.9300537109375,1048.06005859375,1041.2099609375,1042.8399658203125,1040.219970703125,1027.6700439453125,1022.3699951171875,1027.0,1009.5999755859375,1017.9600219726562,1020.2899780273438,1020.969970703125,1020.3099975585938,1015.1500244140625,1010.8200073242188,1005.1199951171875,1003.1699829101562,998.030029296875,994.6300048828125,988.6799926757812,986.4199829101562,991.2100219726562,984.72998046875,982.719970703125,980.9400024414062,990.4299926757812,993.030029296875,999.0900268554688,1000.9299926757812,1003.7999877929688,1012.1699829101562,1016.2000122070312,1021.8499755859375,1013.52001953125,1013.8300170898438,1008.1699829101562,1007.280029296875,1004.27001953125,1001.3499755859375,1002.75,1003.9400024414062,1007.489990234375,1010.0999755859375,1009.5900268554688,1007.6699829101562,1001.7899780273438,997.6099853515625,994.97998046875,998.02001953125,999.7999877929688,1008.75,1021.8699951171875,1032.6700439453125,1036.3299560546875,1056.56005859375,1063.0999755859375,1066.1600341796875,1075.6700439453125,1076.5799560546875,1082.4599609375,1089.3499755859375,1086.7099609375,1083.93994140625,1072.469970703125,1067.8399658203125,1066.93994140625,1066.81005859375,1059.77001953125,1063.6700439453125,1068.06005859375,1069.1500244140625,1068.6500244140625,1065.949951171875,1070.68994140625,1063.0899658203125,1068.3800048828125,1064.8299560546875,1067.06005859375,1069.300048828125,1072.0999755859375,1072.9100341796875,1075.1400146484375,1077.52001953125,1077.72998046875,1080.219970703125,1081.6600341796875,1085.8900146484375,1087.989990234375,1090.18994140625,1093.9000244140625,1098.3599853515625,1101.3199462890625,1105.31005859375,1112.3699951171875,1117.3599853515625,1144.06005859375,1159.93994140625,1171.5699462890625,1187.81005859375,1207.760009765625,1178.699951171875,1203.030029296875,1166.719970703125,1173.0799560546875,1160.449951171875,1157.06005859375,1140.800048828125,1130.530029296875,1120.1600341796875,1109.3699951171875,1108.1199951171875,1103.9100341796875,1107.4100341796875,1111.449951171875,1115.8699951171875,1117.199951171875,1121.4100341796875,1120.3699951171875,1117.6300048828125,1117.18994140625,1112.22998046875,1109.6300048828125,1106.739990234375,1104.0400390625,1106.02001953125,1108.5799560546875,1114.25,1116.4599609375,1115.989990234375,1119.1099853515625,1125.25,1124.8499755859375,1126.1600341796875,1129.0699462890625,1136.75,1142.239990234375,1147.8199462890625,1146.3299560546875,1149.760009765625,1157.31005859375,1172.02001953125,1172.469970703125,1179.2900390625,1186.219970703125,1192.72998046875,1203.739990234375,1204.8299560546875,1209.3499755859375,1222.530029296875,1229.3800048828125,1228.3800048828125,1234.5,1256.6199951171875,1270.489990234375,1292.1800537109375,1318.2900390625,1331.8199462890625,1354.550048828125,1380.8299560546875,1336.9599609375,1347.3199462890625,1387.5400390625,1395.5699462890625,1422.6500244140625,1461.0699462890625,1470.8499755859375,1516.300048828125,1591.18994140625,1603.2900390625,1713.22998046875,1795.72998046875,1809.25,1847.780029296875,1857.6300048828125,1886.4200439453125,1957.3599853515625,2050.159912109375,2247.679931640625,2298.7099609375,2312.179931640625,2357.320068359375,2562.280029296875,2969.68994140625,2893.669921875,3028.60009765625,2989.3701171875,3103.179931640625,3609.530029296875,3721.909912109375,3880.7099609375,3915.8798828125,3924.27001953125,3931.72998046875,3922.679931640625,3909.10009765625,3906.0,3909.409912109375,3617.830078125,3693.219970703125,3478.110107421875,3421.780029296875,3182.280029296875,3028.1298828125,3048.3701171875,3038.31005859375,3125.43994140625,3239.989990234375,3303.2099609375,3373.550048828125,3361.3798828125,3416.81005859375,3573.10009765625,3730.97998046875,3710.25,3704.800048828125,3663.419921875,3690.75,3722.93994140625,3834.260009765625,3829.659912109375,3907.77001953125,3984.949951171875,4150.5,4167.10986328125,4066.6298828125,3945.39990234375,3801.679931640625,3713.18994140625,3753.929931640625,3708.929931640625,3681.719970703125,3651.739990234375,3467.550048828125,3498.820068359375,3425.590087890625,3420.77001953125,3388.8798828125,3372.510009765625,3285.320068359375,3306.5,3319.639892578125,3321.280029296875,3335.139892578125,3345.239990234375,3335.320068359375,3328.409912109375,3283.590087890625,3260.820068359375,3269.510009765625,3305.389892578125,3404.179931640625,3494.7099609375,3610.340087890625,3758.5,3707.419921875,3807.989990234375,3791.969970703125,3859.97998046875,3807.530029296875,3823.89990234375,3912.860107421875,3914.0400390625,3994.97998046875,4083.219970703125,4316.83984375,4422.35986328125,4461.4501953125,4530.490234375,4504.93017578125,4567.240234375,4696.740234375,4695.740234375,4698.68017578125,4753.2001953125,4820.240234375,5072.56005859375,5232.33984375,5305.43994140625,5355.009765625,5456.66015625,5638.080078125,5884.06982421875,6007.18017578125,6504.58984375,6708.31982421875,6579.89013671875,6504.240234375,6667.740234375,6564.669921875,6855.169921875,6919.18994140625,6895.5498046875,6889.72021484375,7496.81982421875,7868.47021484375,8322.2802734375,8464.26953125,9060.9697265625,8735.169921875,8591.3896484375,8296.66015625,7896.830078125,7762.02001953125,7484.080078125,7539.7900390625,7609.1201171875,7663.759765625,7678.89013671875,7666.009765625,7643.0400390625,7409.39013671875,7010.47998046875,7042.56005859375,7046.35009765625,7232.89990234375,7527.3798828125,7671.68994140625,7777.81982421875,7661.60986328125,7590.509765625,7666.02001953125,7728.2099609375,7692.47021484375,7653.240234375,7502.0498046875,7402.60009765625,7041.6298828125,6976.06005859375,6918.89990234375,6940.10009765625,7029.16015625,7027.77978515625,7111.1201171875,7163.169921875,7356.56982421875,7352.0,7353.3798828125,7340.7998046875,7457.08984375,7508.10986328125,7719.35986328125,7952.080078125,8176.419921875,8138.3798828125,8302.08984375,8250.25,8278.349609375,8391.9501953125,8315.7998046875,8218.5302734375,8167.2099609375,8100.60009765625,8105.43017578125,8214.330078125,8196.8095703125,8350.830078125,8215.3603515625,8192.33984375,8066.14013671875,8027.740234375,7996.77001953125,7966.64013671875,7966.27978515625,8009.85986328125,8073.4501953125,8000.75,7829.6201171875,7874.60986328125,7649.35986328125,7632.08984375,7627.08984375,7756.27978515625,7767.85009765625,7917.919921875,8016.97021484375,8187.02001953125,8241.0595703125,8358.490234375,8394.740234375,8331.2197265625,8371.1904296875,8308.330078125,8324.990234375,8314.650390625,8272.259765625,8223.3203125,8209.5302734375,8207.4296875,8193.41015625,8197.1796875,8199.6904296875,8197.3603515625,8195.7802734375,8235.48046875,8256.6796875,8663.6201171875,8741.7001953125,9130.76953125,9236.580078125,9412.490234375]
    # with_out_n_steps_point = len(actuals) - int(Config.n_steps)
    # actuals = actuals[:with_out_n_steps_point]
    # print(type(actuals), type(m_ann), type(m_b_ann))
    # train_point = round(len(actuals) - (len(actuals) * Config.test_size))
    # print(len(actuals), train_point)
    # actuals_test = actuals[train_point:]
    # m_ann_test = m_ann[train_point:]
    # m_b_ann_test = m_b_ann[train_point:]
    # print(len(m_ann_test), len(m_b_ann_test), len(actuals_test))
    # print(Evaluation.calculateMetricsUnivariate(np.array(actuals_test), np.array(m_ann_test)))
    # m_ann_test_mae = mean_absolute_error(actuals_test, m_ann_test)
    # m_b_ann_test_mae = mean_absolute_error(actuals_test, m_b_ann_test)
    # print(m_ann_test_mae, m_b_ann_test_mae)
    # ensemble = (np.array(m_ann_test) + np.array(m_b_ann_test)) / 2
    # mae = mean_absolute_error(actuals_test, ensemble)
    # print(mae)
    # y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(5)]
    # 100 samples, 2 features
    # y2 = np.random.rand(2, 5).tolist()  # 100 samples, 2 features
    # print(y, y2)
    # X = Helper.merge_and_clean(2, arr1=y, arr2=y2)
    # print(X)
    # data_mapping = {
    #     index + 1: {"x": x, "y": y}
    #     for index, (x, y) in enumerate(zip(X, y))
    # }
    # print(data_mapping)
    # y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification task
    # print('x : ', X)
    # print('x sh: ', X.shape)
    # print('y : ', y)
    # print('y sh: ', y.shape)
    # print('y out : ', np.array(y_outputs))
    # print('x sh: ', X.shape)
    # print('y out sh: ', np.array(y_outputs).shape)
    # Split the dataset into training and testing sets
    # The test_size parameter specifies the proportion of the dataset to include in the test split.
    # The random_state parameter ensures reproducibility by fixing the random seed.
    # X_train, X_test, y_train, y_test = train_test_split(X, np.squeeze(y_outputs), test_size=0.2, shuffle=False, random_state=42)
    # print('X_train : ', X_train)
    # print('X_train sh: ', X_train.shape)
    # print('X_test : ', X_test)
    # print('X_test sh: ', X_test.shape)
    # print('y_train : ', y_train)
    # print('y_train sh: ', y_train.shape)
    # print('y_test : ', y_test)
    # print('y_test sh: ', y_test.shape)
    # Now, X_train and y_train contain the training data, and X_test and y_test contain the testing data.
    # You can use these datasets to train and evaluate your machine learning model.

    # array1 = ['cmm', 2, 3, 4, 5]
    # array2 = [4, 'cmm', 6, 7, 8]
    #
    # # Merge arrays and remove duplicates
    # merged_array = list(set(array1 + array2))
    #
    # # Print the merged array without duplicates
    # print(merged_array)
