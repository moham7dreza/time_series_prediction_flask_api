from pandora.Config.Config import Config
from pandora.Data.DataSampler import DataSampler
from pandora.SequentialModelBuilder import ModelBuilder
from pandora.Helper.Helper import Helper


class Univariate:
    @staticmethod
    def univariant_series(model_name, train, test):
        # split into samples
        X, y = DataSampler.split_sequences(Config.univariate, train)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # Define the path for saving/loading the model
        model_path = Config.drive_model_folder_path + '/{}.h5'.format('U-' + model_name)

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
