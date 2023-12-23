from src.Config.Config import Config
from src.Data.DataSampler import DataSampler
import os
from src.Helper.Helper import Helper
from src.Model.ModelBuilder import ModelBuilder
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split


class Univariate:
    @staticmethod
    def univariant_series(model_name, train, test):
        # split into samples
        X, y = DataSampler.split_sequences(Config.univariate, train)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # Define the path for saving/loading the model
        if Config.colab:
            model_path = Config.drive_model_folder_path + '/{}.h5'.format('U-' + model_name)
        else:
            model_path = Config.local_model_folder_path + '/{}.h5'.format('U-' + model_name)

        # Check if the model file exists
        if os.path.exists(model_path):
            # Load the existing model
            model = load_model(model_path)
            print("Model '{}' loaded from file.".format('U-' + model_name))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            model.fit(X, y, epochs=Config.epochs_for_univariate_series, verbose=0)

            # Save the model
            model.save(model_path)
            print("Model '{}' saved to file.".format('U-' + model_name))

        # demonstrate prediction
        x_input = test.reshape((1, Config.n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print('U-' + model_name + ' : ', yhat)
        return yhat

    @staticmethod
    def splitted_univariate_series(model_name, dataset, scaler):
        # split into samples
        X, y = DataSampler.split_sequences(Config.univariate, dataset)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the path for saving/loading the model
        if Config.colab:
            model_path = Config.drive_model_folder_path + '/{}.h5'.format('U-' + model_name)
        else:
            model_path = Config.local_model_folder_path + '/{}.h5'.format('U-' + model_name)

        # Check if the model file exists
        if Config.checkForModelExistsInFolder and os.path.exists(model_path):
            # Load the existing model
            model = load_model(model_path)
            print("Model '{}' loaded from file.".format('U-' + model_name))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            model.fit(X_train, y_train, epochs=Config.epochs_for_univariate_series, batch_size=32)

            # Save the model
            model.save(model_path)
            print("Model '{}' saved to file.".format('U-' + model_name))

        loss = model.evaluate(X_test, y_test)
        print("Model '{}' loss is : .".format('U-' + model_name), loss)

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

        return {
            'actual': Helper.merge_and_clean(round_decimals=2, arr1=y_train, arr2=y_test),
            'predictions': Helper.merge_and_clean(round_decimals=2, arr1=train_predictions, arr2=test_predictions)
        }
