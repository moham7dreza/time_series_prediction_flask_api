import os
from src.Config.Config import Config
from src.Data.DataSampler import DataSampler
from src.Helper.Helper import Helper
from src.Model.ModelBuilder import ModelBuilder
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np


class Multivariate:
    @staticmethod
    def multivariate_multiple_output_parallel_series(model_name, train, test):
        # convert into input/output
        X, y = DataSampler.split_sequences(Config.multivariate, train)
        # the dataset knows the number of features, e.g., 2
        n_features = X.shape[2]

        # separate output dynamically based on n_features
        y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_features)]

        # Define the path for saving/loading the model
        if Config.colab:
            model_path = Config.drive_model_folder_path + '/{}.h5'.format('M-' + model_name)
        else:
            model_path = Config.local_model_folder_path + '/{}.h5'.format('M-' + model_name)
        # Check if the model file exists
        if os.path.exists(model_path):
            # Load the existing model
            model = load_model(model_path)
            print("Model '{}' loaded from file.".format('M-' + model_name))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            model.fit(X, y_outputs, epochs=Config.epochs_for_multivariate_series, verbose=0)

            # Save the model
            model.save(model_path)
            print("Model '{}' saved to file.".format('M-' + model_name))

        # demonstrate prediction
        x_input = test.reshape((1, Config.n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print('M-' + model_name + ' : ', yhat)
        return yhat

    @staticmethod
    def splitted_multivariate_series(model_name, dataset, scaler, dates):
        # convert into input/output
        X, y = DataSampler.split_sequences(Config.multivariate, dataset)
        # the dataset knows the number of features, e.g., 2
        n_features = X.shape[2]
        # print('y sh : ', y.shape)  # (284, 5)
        # print('y : ', y)
        # separate output dynamically based on n_features
        y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_features)]
        # print('X sh , y_outputs sh : ', X.shape, np.array(y_outputs).shape)  # (284, 3, 5) (5, 284, 1)
        # print('X : ', X.tolist())
        # print('y_out : ', np.array(y_outputs).tolist())
        # print('y_out np  ', np.array(y_outputs))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.test_size,
                                                            random_state=Config.random_state)

        # Define the path for saving/loading the model
        if Config.colab:
            model_path = Config.drive_model_folder_path + '/{}.h5'.format('M-' + model_name)
        else:
            model_path = Config.local_model_folder_path + '/{}.h5'.format('M-' + model_name)

        # Check if the model file exists
        if Config.checkForModelExistsInFolder and os.path.exists(model_path):
            # Load the existing model
            model = load_model(model_path)
            print("Model '{}' loaded from file.".format('M-' + model_name))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            model.fit(X_train, y_train, epochs=Config.epochs_for_multivariate_series, verbose=0)

            # Save the model
            model.save(model_path)
            print("Model '{}' saved to file.".format('M-' + model_name))

        loss = model.evaluate(X_test, y_test)
        print("Model '{}' loss is : .".format('M-' + model_name), loss)

        # Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        # print('eval : ', np.array(train_predictions).shape, np.array(test_predictions).shape)  # (5, 227, 1) (5,
        # 57, 1)

        # Transform predictions back to original scale
        train_predictions = np.concatenate([scaler.inverse_transform(pred) for pred in train_predictions], axis=1)
        y_train = scaler.inverse_transform(y_train)
        test_predictions = np.concatenate([scaler.inverse_transform(pred) for pred in test_predictions], axis=1)
        y_test = scaler.inverse_transform(y_test)

        # Calculate errors
        train_rmse = np.sqrt(np.mean(np.square(train_predictions - y_train)))
        test_rmse = np.sqrt(np.mean(np.square(test_predictions - y_test)))

        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")

        actuals = Helper.merge_and_clean(round_decimals=2, arr1=y_train, arr2=y_test),
        predictions = Helper.merge_and_clean(round_decimals=2, arr1=train_predictions, arr2=test_predictions)
        print('dates : ', len(dates))
        print('actuals : ', len(actuals))
        print('predictions : ', len(predictions))

        # Check if all arrays have the same length
        if len(dates) == len(actuals) == len(predictions):
            # Create the mapping
            data_mapping = {
                index + 1: {"date": date, "actual": actual, "predict": predict}
                for index, (date, actual, predict) in enumerate(zip(dates, actuals, predictions))
            }
            return data_mapping
        else:
            raise ValueError("Arrays must have the same length.")
