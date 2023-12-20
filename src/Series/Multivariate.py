from pandora.Config.Config import Config
from pandora.Data.DataSampler import DataSampler
from pandora.Helper.Helper import Helper
from pandora.Model.ModelBuilder import ModelBuilder
import os

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
        model_path = Config.drive_model_folder_path + '/{}.h5'.format('M-' + model_name)

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