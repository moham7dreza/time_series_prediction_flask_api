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
from src.Helper.Helper import Helper

if __name__ == '__main__':
    # from sklearn.model_selection import train_test_split

    # Assume you have a dataset with features (X) and labels (y)
    # X should be a 2D array (or DataFrame), and y should be a 1D array (or Series)
    # Let's generate a sample dataset for demonstration purposes.

    # Import necessary libraries
    # import numpy as np

    # Generate a synthetic dataset
    # X = np.round(np.random.rand(10, 3, 5), 2)  # 100 samples, 3 features
    # X2 = np.random.rand(10, 3, 5)  # 100 samples, 3 features
    # y = np.round(np.random.rand(10, 5), 2)
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

    array1 = ['cmm', 2, 3, 4, 5]
    array2 = [4, 'cmm', 6, 7, 8]

    # Merge arrays and remove duplicates
    merged_array = list(set(array1 + array2))

    # Print the merged array without duplicates
    print(merged_array)


