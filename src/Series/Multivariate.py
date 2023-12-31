import os
from src.Config.Config import Config
from src.Data.DataSampler import DataSampler
from src.Helper.Helper import Helper
from src.Model.Evaluation import Evaluation
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
    def splitted_multivariate_series(model_name, dataset, scaler, dates, titles):
        # print("title : ", titles)
        # print("------------------------------------------------------")
        # print("dataset shape, type : ", dataset.shape, type(dataset))
        # print("dataset : ", dataset)
        # print("------------------------------------------------------")
        # print("dates len and type : ", len(dates), type(dates))  # <class 'list'>
        # print("dates : ", np.array(dates))
        # print("------------------------------------------------------")
        # convert into input/output
        X, y = DataSampler.split_sequences(Config.multivariate, dataset)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.test_size,
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
            # Fit model TODO X or X_trait?
            model.fit(X_train, y_outputs, epochs=Config.epochs_for_multivariate_series, verbose=0)

            # Save the model
            model.save(model_path)
            print("Model '{}' saved to file.".format('M-' + model_name))
        # print("------------------------------------------------------")
        y_test_outputs = [y_test[:, i].reshape((y_test.shape[0], 1)) for i in range(n_features)]
        # print('y_test_outputs shape and type : ', np.array(y_test_outputs).shape, type(y_test_outputs))  # (5, 284, 1)
        # print('y_test_outputs : ', np.array(y_test_outputs))
        loss = model.evaluate(X_test, y_test_outputs)
        # print("Model '{}' loss is : ".format('M-' + model_name), loss)
        # print("------------------------------------------------------")
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

        # Sum the arrays element-wise at the same index
        predictions = [np.concatenate(eachDatasetPrediction) for eachDatasetPrediction in
                       zip(train_predictions, test_predictions)]
        predictions = np.round(np.squeeze(predictions), 2)
        actuals = np.concatenate((y_train, y_test))
        actuals = [actuals[:, i].reshape((actuals.shape[0], 1)) for i in range(n_features)]
        actuals = np.round(np.squeeze(actuals), 2)
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
            title: {"actual": actual.tolist(), "predict": predict.tolist()}
            for title, actual, predict in
            zip(titles, actuals, predictions)
        }
        metrics = {
            title: {metric: test_metrics[metric][index] for metric in test_metrics}
            for index, title in enumerate(titles)
        }

        return results, metrics
