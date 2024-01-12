import os

import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split

from src.Config.Config import Config
from src.Data.DataSampler import DataSampler
from src.Helper.Helper import Helper
from src.Model.Evaluation import Evaluation
from src.Model.ModelBuilder import ModelBuilder


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
            # print("Model '{}' loaded from file.".format('M-' + model_name))
        else:
            # Define model
            model = ModelBuilder.getModel(model_name, n_features)
            # Fit model
            model.fit(X, y_outputs, epochs=Config.epochs_for_multivariate_series, batch_size=Config.batch_size,
                      verbose=0)

            # Save the model
            model.save(model_path)
            # print("Model '{}' saved to file.".format('M-' + model_name))

        # demonstrate prediction
        x_input = test.reshape((1, Config.n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print('M-' + model_name + ' : ', yhat)
        return yhat

    @staticmethod
    def splitted_multivariate_series(model_name, dataset, scaler, titles, price, PredictionDTO,
                                     fit_regressor=False):
        price = Helper.str_remove_flags(price)
        # print("title : ", titles)
        # print("------------------------------------------------------")
        # print("dataset shape, type : ", dataset.shape, type(dataset))
        # print("dataset : ", dataset)
        # print("------------------------------------------------------")
        # print("dates len and type : ", len(dates), type(dates))  # <class 'list'>
        # print("dates : ", np.array(dates))
        # print("------------------------------------------------------")
        # convert into input/output
        X, y = DataSampler.split_sequences(Config.multivariate, dataset, PredictionDTO.n_steps)
        # print("X ,y shape and type : ", X.shape, y.shape, type(X), type(y))  # (284, 3, 5) (284, 5)
        # the dataset knows the number of features
        n_features = X.shape[2]
        # print('X : ', X)
        # print('y : ', y)
        # print("------------------------------------------------------")
        # separate output dynamically based on n_features
        y_outputs = [y[:, i].reshape((y.shape[0], 1)) for i in range(n_features)]
        # print('y_outputs shape and type : ', np.array(y_outputs).shape, type(y_outputs))  # (5, 284, 1)
        # print('y_outputs : ', np.array(y_outputs))
        # print("------------------------------------------------------")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PredictionDTO.test_size,
                                                            random_state=Config.random_state, shuffle=False)
        # print("X_train, X_test, y_train, y_test type : ", type(X_train), type(X_test), type(y_train), type(y_test))
        # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print("X_train, X_test, y_train, y_test shape : ", X_train.shape, X_test.shape, y_train.shape,
        #       y_test.shape)  # (227, 3, 5) (57, 3, 5) (227, 5) (57, 5)
        # print("X_train : ", X_train)
        # print("y_train : ", y_train)
        # print("X_test : ", X_test)
        # print("y_test : ", y_test)
        # print("------------------------------------------------------")
        # Define the path for saving/loading the model
        savedModelName = Multivariate.extract_saved_model_name(PredictionDTO, price, model_name)

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
            model = ModelBuilder.getModel(model_name, n_features, PredictionDTO.n_steps, PredictionDTO.dropout_rate)
            # Fit model TODO X or X_trait?
            if fit_regressor:
                model.fit(X, y_outputs)
            else:
                model.fit(X, y_outputs, epochs=PredictionDTO.multivariate_epochs, batch_size=PredictionDTO.batch_size)

            # Save the model
            model.save(model_path)
            # print("Model '{}' saved to file.".format(savedModelName))
        # print("------------------------------------------------------")
        y_test_outputs = [y_test[:, i].reshape((y_test.shape[0], 1)) for i in range(n_features)]
        # print('y_test_outputs shape and type : ', np.array(y_test_outputs).shape, type(y_test_outputs))  # (5, 284, 1)
        # print('y_test_outputs : ', np.array(y_test_outputs))
        loss = model.evaluate(X_test, y_test_outputs)
        # print("Model '{}' loss is : ".format('M-' + model_name), loss)
        # print("------------------------------------------------------")
        # future predictions

        future_prediction = Multivariate.get_future_predictions(PredictionDTO, dataset, model, n_features)

        # Evaluate the model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        # print("predictions type : ", type(train_predictions), type(test_predictions))  # <class 'list'> <class 'list'>
        # print('predictions shape : ', np.array(train_predictions).shape, np.array(test_predictions).shape)
        # (5, 227, 1) (5, 57, 1)
        # print(type(train_predictions[0]))  # <class 'numpy.ndarray'>
        # print('train predictions : ', np.array(train_predictions))
        # print('test predictions : ', np.array(test_predictions))
        # print("------------------------------------------------------")
        # Transform predictions back to original scale
        # train_predictions = np.concatenate([scaler.inverse_transform(pred) for pred in train_predictions], axis=1)
        # y_train = scaler.inverse_transform(y_train)
        # test_predictions = np.concatenate([scaler.inverse_transform(pred) for pred in test_predictions], axis=1)
        # y_test = scaler.inverse_transform(y_test)
        # print(len(train_predictions), len(test_predictions))  # 5 5

        # print(train_predictions[0].shape, test_predictions[0].shape)  # (227, 1) (57, 1)
        # print(y_train.shape, y_test.shape)  # (227, 5) (57, 5)
        # Calculate errors

        test_metrics = Evaluation.calculateMetricsMultivariate(y_test, test_predictions)
        # print("------------------------------------------------------")
        # print("Train ref RMSE : ", train_rmse_ref)

        actuals, predictions = Multivariate.transform_data(n_features, test_predictions, train_predictions, y_test,
                                                           y_train)
        # print("actuals and predictions type : ", type(actuals),
        #       type(predictions))  # <class 'list'> <class 'numpy.ndarray'>
        # print("actuals and predictions shape : ", actuals.shape, predictions.shape)  # (284, 5)
        # print("actuals : ", actuals)
        # print("predictions : ", predictions)
        # print("------------------------------------------------------")
        # actuals = Helper.merge_and_clean(round_decimals=2, arr1=y_train, arr2=y_test),
        # predictions = Helper.merge_and_clean(round_decimals=2, arr1=train_predictions, arr2=test_predictions)
        # print('dates : ', len(dates))
        # print('actuals : ', len(actuals))
        # print('predictions : ', len(predictions))

        # Check if all arrays have the same length
        # if len(dates) == len(actuals) == len(predictions):
        #     # Create the mapping
        #     data_mapping = {
        #         index + 1: {"date": date, "actual": actual, "predict": predict}
        #         for index, (date, actual, predict) in enumerate(zip(dates, actuals, predictions))
        #     }
        #     return data_mapping
        # else:
        #     raise ValueError("Arrays must have the same length.")
        results = {
            title: {"actual": actual.tolist(), "predict": predict.tolist(), "future_predict": future_predict.tolist()}
            for title, actual, predict, future_predict in
            zip(titles, actuals, predictions, future_prediction)
        }
        metrics = {
            title: {metric: test_metrics[metric][index] for metric in test_metrics}
            for index, title in enumerate(titles)
        }

        return results, metrics

    @staticmethod
    def transform_data(n_features, test_predictions, train_predictions, y_test, y_train):
        # Sum the arrays element-wise at the same index
        predictions = [np.concatenate(eachDatasetPrediction) for eachDatasetPrediction in
                       zip(train_predictions, test_predictions)]
        predictions = np.round(np.squeeze(predictions), 2)
        actuals = np.concatenate((y_train, y_test))
        actuals = [actuals[:, i].reshape((actuals.shape[0], 1)) for i in range(n_features)]
        actuals = np.round(np.squeeze(actuals), 2)
        return actuals, predictions

    @staticmethod
    def get_future_predictions(PredictionDTO, dataset, model, n_features):
        input_data = dataset[-PredictionDTO.n_steps:]
        for day in range(PredictionDTO.n_predict_future_days):
            future_prediction = model.predict(
                input_data[-PredictionDTO.n_steps:].reshape((1, PredictionDTO.n_steps, n_features)))
            input_data = np.vstack((input_data, np.squeeze(future_prediction)))
        future_prediction = input_data[PredictionDTO.n_steps:]
        future_prediction = [future_prediction[:, i].reshape((future_prediction.shape[0], 1)) for i in
                             range(n_features)]
        future_prediction = np.round(np.squeeze(future_prediction), 2)
        return future_prediction

    @staticmethod
    def extract_saved_model_name(PredictionDTO, price, model_name):
        savedModelName = (
                'M-' + model_name + '-' + price
                + '-[' + PredictionDTO.start_date.replace('-', '')
                + '-' + PredictionDTO.end_date.replace('-', '')
                + ']-'
                + str(PredictionDTO.n_steps) + '-steps-'
                + str(int(PredictionDTO.test_size * 100)) + '%-test-'
                + str(PredictionDTO.multivariate_epochs) + '-epochs-'
                + str(PredictionDTO.batch_size) + '-batches-'
                + str(int(PredictionDTO.dropout_rate * 100)) + '%-dropout'
        )
        return savedModelName
