from src.Config.Config import Config
from src.Data.DataSampler import DataSampler
import os
from src.Helper.Helper import Helper
from src.Model.Evaluation import Evaluation
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
    def splitted_univariate_series(model_name, dataset, scaler, dates, label, fit_regressor=False):
        label = Helper.str_remove_flags(label)
        # print(len(dates), len(dataset))  # 284 287 date = dataset with out last 3 steps
        # print("dataset shape, type : ", dataset.shape, type(dataset))  # (287, 1) <class 'numpy.ndarray'>
        # print("dataset : ", dataset)
        # split into samples
        X, y = DataSampler.split_sequences(Config.univariate, dataset)
        # print("X ,y shape and type : ", X.shape, y.shape, type(X), type(y))  # (284, 3, 1) (284, 1) <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print('X : ', X)
        # print('y : ', y)
        # print("------------------------------------------------------")

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.test_size,
                                                            random_state=Config.random_state, shuffle=False)
        # print("X_train, X_test, y_train, y_test type : ", type(X_train), type(X_test), type(y_train), type(y_test))
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print("X_train, X_test, y_train, y_test shape : ", X_train.shape, X_test.shape, y_train.shape,
        #       y_test.shape)  # (227, 3, 1) (57, 3, 1) (227, 1) (57, 1)
        # print("X_train : ", X_train)
        # print("y_train : ", y_train)
        # print("X_test : ", X_test)
        # print("y_test : ", y_test)
        # print("------------------------------------------------------")
        # Define the path for saving/loading the model
        savedModelName = 'U-' + model_name + '-' + label
        if Config.colab:
            model_path = Config.drive_model_folder_path + '/{}.h5'.format(savedModelName)
        else:
            model_path = Config.local_model_folder_path + '/{}.h5'.format(savedModelName)

        # Check if the model file exists
        if Config.checkForModelExistsInFolder and os.path.exists(model_path):
            # Load the existing model
            model = load_model(model_path)
            # print("Model '{}' loaded from file.".format(savedModelName))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            if fit_regressor:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, epochs=Config.epochs_for_univariate_series, batch_size=32)

            # Save the model
            model.save(model_path)
            # print("Model '{}' saved to file.".format(savedModelName))

        loss = model.evaluate(X_test, y_test)
        # print("Model '{}' loss is : ".format(savedModelName), loss)

        # Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Transform predictions back to original scale
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train = scaler.inverse_transform(y_train)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test = scaler.inverse_transform(y_test)

        # print("predictions type : ", type(train_predictions), type(test_predictions))  # <class 'numpy.ndarray'>
        # <class 'numpy.ndarray'>
        # print('predictions shape : ', train_predictions.shape, test_predictions.shape)  # (227, 1) (57, 1)
        # print('train predictions : ', train_predictions)
        # print('test predictions : ', test_predictions)
        # print('pred - actual train : ', train_predictions - y_train)
        # print("------------------------------------------------------")
        # Calculate errors
        # train_metrics = Evaluation.calculateMetricsUnivariate(y_train, train_predictions)
        test_metrics = Evaluation.calculateMetricsUnivariate(y_test, test_predictions)

        actuals = np.round(np.concatenate((y_train, y_test), axis=0), 2)
        predictions = np.round(np.concatenate((train_predictions, test_predictions), axis=0), 2)
        # print('actuals, predictions shape : ', actuals.shape, predictions.shape)  # (284, 1) (284, 1)
        actuals = actuals.squeeze().tolist()
        predictions = predictions.squeeze().tolist()
        # print('actuals : ', actuals)
        # print('predictions : ', predictions)
        # print(len(dates), len(actuals[0]), len(predictions))
        # # Check if all arrays have the same length
        # if len(dates) == len(actuals) == len(predictions):
        #     # Create the mapping
        #     data_mapping = {
        #         index + 1: {"date": date, "actual": actual, "predict": predict}
        #         for index, (date, actual, predict) in enumerate(zip(dates, actuals, predictions))
        #     }
        #     return data_mapping
        # else:
        #     raise ValueError("Arrays must have the same length.")

        # Create the mapping
        # return {
        #     index + 1: {"date": date, "actual": actual, "predict": predict}
        #     for index, (date, actual, predict) in enumerate(zip(dates, actuals[0], predictions))
        # }

        return actuals, predictions, test_metrics
