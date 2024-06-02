import os
from itertools import combinations

import pandas as pd
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, GRU, Dropout, SimpleRNN, \
    ConvLSTM2D
from keras.models import Model
from keras.models import load_model
from numpy import array, hstack, round, concatenate, vstack, squeeze, float32, sqrt, mean
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class Config:
    colab = False
    plotting = False
    date_col = "<DTYYYYMMDD>"
    # start_date = '2017-06-10'
    start_date = '2014-01-07'
    end_date = '2022-05-16'
    # end_date = '2023-12-05'
    n_steps = 3
    epochs_for_multivariate_series = 100
    epochs_for_univariate_series = 100
    dropout_rate = 0.2
    batch_size = 32
    prediction_col = '<CLOSE>'
    local_csv_folder_path = './iran_stock'
    drive_csv_folder_path = '/content/drive/My Drive/iran_stock'
    drive_model_folder_path = '/content/drive/My Drive/time_series_models'
    local_model_folder_path = './time_series_models'
    # 1. set csv dataset file names
    # dollar_file_name = 'dollar_tjgu_from_2012.csv'
    dollar_file_name = 'Dollar_tjgu_new_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    am_car_file_name = 'Iran.Kh..A..M._from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    housing_file_name = 'Housing.Inv.from_2004.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'
    kh_info_file_name = 'Kharazmi.Info._TECH_from_2014.csv'
    car_shargh_file_name = 'E..Kh..Shargh_from_2004.csv'
    # dataset titles
    Dollar = 'Dollar'
    Home = 'Home'
    Oil = 'Oil'
    Car = 'Car'
    Gold = 'Gold'
    Tech = 'Tech'

    datasets_name = [Dollar, Home, Oil, Car, Tech]
    # model names
    CNN = 'CNN'
    LSTM = 'LSTM'
    bi_LSTM = 'B_LSTM'
    GRU = 'GRU'
    bi_GRU = 'B_GRU'
    ANN = 'ANN'
    bi_ANN = 'B_ANN'
    RNN = 'RNN'
    bi_RNN = 'B_RNN'
    RF_REGRESSOR = 'RF_Regressor'
    GB_REGRESSOR = 'GB_Regressor'
    DT_REGRESSOR = 'DT_Regressor'
    XGB_REGRESSOR = 'XGB_Regressor'
    Linear_REGRESSION = 'Linear_REGRESSION'
    Conv_LSTM = 'Conv_LSTM'
    models_name = [
        CNN,
        LSTM,
        bi_LSTM,
        GRU,
        bi_GRU,
        ANN,
        bi_ANN,
        RNN,
        bi_RNN,
    ]
    # series type
    univariate = 'univariate'
    multivariate = 'multivariate'
    series_name = [univariate, multivariate]
    base_project_path = os.path.abspath(os.path.dirname(__file__))
    plot_labels = ['M-CNN', 'U-CNN',
                   'M-B-LSTM', 'U-B-LSTM', 'M-LSTM', 'U-LSTM',
                   'M-GRU', 'U-GRU', 'M-B-GRU', 'U-B-GRU',
                   'M-ANN', 'U-ANN', 'M-B-ANN', 'U-N-ANN',
                   'M-RNN', 'U-RNN', 'M-B-RNN', 'U-N-RNN',
                   'REAL']
    checkForModelExistsInFolder = True
    test_size = 0.2
    random_state = 42
    estimators = 100
    learning_rate = 0.1

    Close = '<CLOSE>'
    High = '<HIGH>'
    Open = '<OPEN>'
    Low = '<LOW>'
    prices_name = [Close, High, Open, Low]

    MAE = 'MAE'
    MSE = 'MSE'
    RMSE = 'RMSE'
    MAPE = 'MAPE'
    R2 = 'R2'
    metrics_name = [MAE, MSE, RMSE, MAPE, R2]

    n_subsequences = 2

    active_datasets = {
        Dollar: dollar_file_name,
        Car: car_shargh_file_name,
        Tech: kh_info_file_name,
        Home: housing_file_name,
        Oil: oil_file_name
    }

    check_for_datasets_date_range = False


class Helper:
    @staticmethod
    def flatten_arr(result):
        items = array(result).flatten()
        return [round(item, 2) for item in items]

    # Convert NumPy float32 to Python float
    @staticmethod
    def convert_to_python_float(obj):
        if isinstance(obj, float32):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [Helper.convert_to_python_float(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: Helper.convert_to_python_float(value) for key, value in obj.items()}
        return obj

    @staticmethod
    def str_remove_flags(string):
        return string.replace('<', '').replace('>', '')

    @staticmethod
    def flatten_list(nested_list):
        flat_list = []
        for sublist in nested_list:
            if isinstance(sublist, list):
                flat_list.extend(Helper.flatten_list(sublist))
            else:
                flat_list.append(sublist)
        return flat_list


class PredictionDTO:
    def __init__(self):
        self.n_steps = 3
        self.datasets = ['Dollar']
        self.models = ['CNN', 'LSTM']
        self.series = ['univariate']
        self.prices = ['<CLOSE>']
        self.metrics = ['MAE']
        self.n_top_models_to_ensemble = 0
        self.apply_combinations = False
        self.n_predict_future_days = 14
        self.start_date = '2014-01-07'
        self.end_date = '2022-05-16'
        self.univariate_epochs = 100
        self.multivariate_epochs = 500
        self.batch_size = 32
        try:
            self.test_size = 20 / 100
            self.dropout_rate = 20 / 100
        except (TypeError, ValueError):
            self.test_size = None
            self.dropout_rate = None


class DataLoader:

    @staticmethod
    def read_csv_file_from_local(file_name):
        folder_path = Config.local_csv_folder_path
        # Get a list of all CSV files in the folder
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        for file in csv_files:
            if file_name in file:
                file_path = os.path.join(folder_path, file)
                return pd.read_csv(file_path)

    @staticmethod
    def data_preprocessing(dataset, PredictionDTO, date_col=Config.date_col,
                           format=True):
        # sort by date
        dataset = dataset.sort_values(by=date_col)
        # print(dataset)

        if format:
            # Assuming 'date' is the column containing date in integer format
            dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
        else:
            dataset[date_col] = pd.to_datetime(dataset[date_col], format='mixed')

        # Convert object columns to strings
        object_columns = dataset.select_dtypes(include='object').columns
        dataset[object_columns] = dataset[object_columns].astype(str)

        # print(dataset.dtypes)

        # Identify and exclude object columns
        non_object_columns = dataset.select_dtypes(exclude='object').columns
        # Create a new DataFrame without object columns
        dataset = dataset[non_object_columns]

        # print(dataset)
        dataset = dataset.set_index(date_col)
        # print(dataset.dtypes)

        if Config.check_for_datasets_date_range:
            # Check if the date range is present in the DataFrame
            date_range_present = ((dataset.index.min() <= pd.to_datetime(PredictionDTO.start_date))
                                  & (dataset.index.max() >= pd.to_datetime(PredictionDTO.end_date)))
            if not date_range_present:
                raise ValueError('selected date range not exists')

        dataset = dataset.resample('W-Sat').mean().ffill()
        # print(dataset)

        dataset = dataset.loc[PredictionDTO.start_date:PredictionDTO.end_date]
        # print(dataset)

        return dataset

    @staticmethod
    def get_datasets_refactored(PredictionDTO):
        datasets = {}

        # 1. Get all active datasets
        for dataset_name, dataset_path in Config.active_datasets.items():
            dataset = None

            # 2. Load dataset based on environment

            dataset = DataLoader.read_csv_file_from_local(dataset_path)

            # 3. Perform data preprocessing
            if dataset_name == Config.Dollar:
                dataset = DataLoader.data_preprocessing(dataset, PredictionDTO, format=False)
            else:
                dataset = DataLoader.data_preprocessing(dataset, PredictionDTO)

            # 4. Add dataset to the dictionary
            datasets[dataset_name] = dataset

        return datasets


class DataSampler:
    # split a multivariate sequence into samples
    @staticmethod
    def split_sequences(series_type, sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences) - 1:
                break
            # gather input and output parts of the pattern
            if series_type == 'multivariate':
                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            elif series_type == 'univariate':
                seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
            else:
                raise Exception('Unknown series type')
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)


class Evaluation:
    @staticmethod
    def calculateMetricsUnivariate(actuals, predictions):
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(actuals, predictions)
        # print("Mean Absolute Error (MAE):", mae)

        # Mean Squared Error (MSE)
        mse = mean_squared_error(actuals, predictions)
        # print("Mean Squared Error (MSE):", mse)

        # Root Mean Squared Error (RMSE)
        rmse = sqrt(mse)
        # print("Root Mean Squared Error (RMSE):", rmse)

        # Mean Absolute Percentage Error (MAPE)
        mape = mean(abs((actuals - predictions) / actuals)) * 100
        # print("Mean Absolute Percentage Error (MAPE):", mape)

        # R-squared (R2)
        r2 = r2_score(actuals, predictions)
        # print("R-squared (R2):", r2)

        return {
            Config.MAE: round(mae, 2),
            Config.MSE: round(mse, 2),
            Config.RMSE: round(rmse, 2),
            Config.MAPE: round(mape, 2),
            Config.R2: round(r2, 2),
        }


class ModelBuilder:
    @staticmethod
    def get_multi_output_stacked_LSTM_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        lstm = LSTM(100, activation='relu', return_sequences=True)(visible)
        lstm = Dropout(dropout_rate)(lstm)
        ##
        lstm = LSTM(100, activation='relu')(lstm)
        lstm = Dropout(dropout_rate)(lstm)
        lstm = Flatten()(lstm)
        lstm = Dense(50, activation='relu')(lstm)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(lstm) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_LSTM_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        bi_lstm = Bidirectional(LSTM(100, activation='relu', return_sequences=True))(visible)
        bi_lstm = Dropout(dropout_rate)(bi_lstm)
        ##
        bi_lstm = Bidirectional(LSTM(100, activation='relu'))(bi_lstm)
        bi_lstm = Dropout(dropout_rate)(bi_lstm)
        bi_lstm = Flatten()(bi_lstm)
        bi_lstm = Dense(50, activation='relu')(bi_lstm)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(bi_lstm) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_GRU_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        gru = GRU(100, activation='relu', return_sequences=True)(visible)
        gru = Dropout(dropout_rate)(gru)
        ##
        gru = GRU(100, activation='relu')(gru)
        gru = Dropout(dropout_rate)(gru)
        gru = Flatten()(gru)
        gru = Dense(50, activation='relu')(gru)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(gru) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_GRU_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        bi_gru = Bidirectional(GRU(100, activation='relu', return_sequences=True))(visible)
        bi_gru = Dropout(dropout_rate)(bi_gru)
        ##
        bi_gru = Bidirectional(GRU(100, activation='relu'))(bi_gru)
        bi_gru = Dropout(dropout_rate)(bi_gru)
        bi_gru = Flatten()(bi_gru)
        bi_gru = Dense(50, activation='relu')(bi_gru)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(bi_gru) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_RNN_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        rnn = SimpleRNN(100, activation='relu', return_sequences=True)(visible)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = SimpleRNN(100, activation='relu')(rnn)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = Flatten()(rnn)
        rnn = Dense(50, activation='relu')(rnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(rnn) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_RNN_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        rnn = Bidirectional(SimpleRNN(100, activation='relu', return_sequences=True))(visible)
        rnn = Dropout(dropout_rate)(rnn)
        ##
        rnn = Bidirectional(SimpleRNN(100, activation='relu'))(rnn)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = Flatten()(rnn)
        rnn = Dense(50, activation='relu')(rnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(rnn) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_ANN_model(n_features, n_steps, dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        ann = Dense(100, activation='relu')(visible)
        ann = Dropout(dropout_rate)(ann)
        ##
        ann = Dense(100, activation='relu')(ann)
        ann = Dropout(dropout_rate)(ann)
        ann = Flatten()(ann)
        ann = Dense(50, activation='relu')(ann)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(ann) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_CNN_model(n_features, n_steps, dropout_rate):
        # define model
        visible = Input(shape=(n_steps, n_features))
        cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(cnn) for _ in range(n_features)]
        # tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_ANN_model(n_features, n_steps, dropout_rate):
        # Define model
        visible = Input(shape=(n_steps, n_features))

        # Bidirectional RNN layer with dropout
        rnn = Bidirectional(SimpleRNN(50, activation='relu', return_sequences=True))(visible)
        rnn = Dropout(dropout_rate)(rnn)

        # Flatten layer to connect with dense layers
        flattened_rnn = Flatten()(rnn)

        # Dense layers with dropout
        dense1 = Dense(64, activation='relu')(flattened_rnn)
        dense1 = Dropout(dropout_rate)(dense1)

        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(dropout_rate)(dense2)

        # Define outputs dynamically based on n_features
        outputs = [Dense(1)(dense2) for _ in range(n_features)]

        # Tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        return model

    @staticmethod
    def get_RF_Regressor_model():
        return RandomForestRegressor(n_estimators=Config.estimators, random_state=Config.random_state)

    @staticmethod
    def get_GB_Regressor_model():
        return GradientBoostingRegressor(n_estimators=Config.estimators, random_state=Config.random_state)

    @staticmethod
    def get_linear_regression_model():
        return LinearRegression()

    @staticmethod
    def get_xgb_regressor_model():
        return XGBRegressor(n_estimators=Config.estimators, learning_rate=Config.learning_rate,
                            random_state=Config.random_state)

    @staticmethod
    def get_dt_regressor_model():
        return DecisionTreeRegressor(random_state=Config.random_state)

    @staticmethod
    def get_Conv_LSTM_model(n_features, n_steps, dropout_rate,
                            n_seq=Config.n_subsequences):
        # define model
        visible = Input(shape=(n_seq, 1, n_steps, n_features))
        cnn = ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu')(visible)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(cnn) for _ in range(n_features)]
        # tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def getModel(model_name, n_features, n_steps, dropout_rate):
        if model_name == Config.CNN:
            model = ModelBuilder.get_multi_output_CNN_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.LSTM:
            model = ModelBuilder.get_multi_output_stacked_LSTM_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.bi_LSTM:
            model = ModelBuilder.get_multi_output_bi_LSTM_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.RNN:
            model = ModelBuilder.get_multi_output_RNN_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.bi_RNN:
            model = ModelBuilder.get_multi_output_bi_RNN_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.GRU:
            model = ModelBuilder.get_multi_output_GRU_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.bi_GRU:
            model = ModelBuilder.get_multi_output_bi_GRU_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.ANN:
            model = ModelBuilder.get_multi_output_ANN_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.bi_ANN:
            model = ModelBuilder.get_multi_output_bi_ANN_model(n_features, n_steps, dropout_rate)
        elif model_name == Config.RF_REGRESSOR:
            model = ModelBuilder.get_RF_Regressor_model()
        elif model_name == Config.GB_REGRESSOR:
            model = ModelBuilder.get_GB_Regressor_model()
        elif model_name == Config.XGB_REGRESSOR:
            model = ModelBuilder.get_xgb_regressor_model()
        elif model_name == Config.DT_REGRESSOR:
            model = ModelBuilder.get_dt_regressor_model()
        elif model_name == Config.Linear_REGRESSION:
            model = ModelBuilder.get_linear_regression_model()
        elif model_name == Config.Conv_LSTM:
            model = ModelBuilder.get_Conv_LSTM_model(n_features, n_steps, dropout_rate)
        else:
            raise Exception("model name not recognized")
        return model


class Univariate:

    @staticmethod
    def splitted_univariate_series(model_name, dataset, scaler, label, PredictionDTO,
                                   fit_regressor=False):
        label = Helper.str_remove_flags(label)

        X, y = DataSampler.split_sequences(Config.univariate, dataset, PredictionDTO.n_steps)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PredictionDTO.test_size,
                                                            random_state=Config.random_state, shuffle=False)

        savedModelName = Univariate.extract_saved_model_name(PredictionDTO, label, model_name)

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
            # Fit model
            if fit_regressor:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, epochs=PredictionDTO.univariate_epochs, batch_size=PredictionDTO.batch_size)

            # Save the model
            model.save(model_path)
            # print("Model '{}' saved to file.".format(savedModelName))

        loss = model.evaluate(X_test, y_test)
        # print("Model '{}' loss is : ".format(savedModelName), loss)

        # future predictions
        future_prediction = Univariate.get_future_predictions(PredictionDTO, dataset, model, n_features, scaler)

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

        actuals, predictions = Univariate.transform_data(test_predictions, train_predictions, y_test, y_train)
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

        return actuals, predictions, test_metrics, future_prediction

    @staticmethod
    def transform_data(test_predictions, train_predictions, y_test, y_train):
        # y_train shape : (227, 1),  y_test shape : (57, 1) -> merge shape (284, 1)
        actuals = round(concatenate((y_train, y_test), axis=0), 2)
        # (227, 1) (57, 1) -> merge shape (284, 1)
        predictions = round(concatenate((train_predictions, test_predictions), axis=0), 2)
        # print('actuals, predictions shape : ', actuals.shape, predictions.shape)  # (284, 1) (284, 1)
        actuals = actuals.squeeze().tolist()
        predictions = predictions.squeeze().tolist()
        return actuals, predictions

    @staticmethod
    def get_future_predictions(PredictionDTO, dataset, model, n_features, scaler):
        if PredictionDTO.n_predict_future_days > 0:
            input_data = dataset[-PredictionDTO.n_steps:]
            for day in range(PredictionDTO.n_predict_future_days):
                future_prediction = model.predict(
                    input_data[-PredictionDTO.n_steps:].reshape((1, PredictionDTO.n_steps, n_features)))
                input_data = vstack((input_data, future_prediction))
            future_prediction = scaler.inverse_transform(input_data[PredictionDTO.n_steps:])
            future_prediction = round(squeeze(future_prediction), 2).tolist()
        else:
            future_prediction = []
        return future_prediction

    @staticmethod
    def extract_saved_model_name(PredictionDTO, label, model_name):
        savedModelName = (
                'U-' + model_name + '-' + label
                + '-[' + PredictionDTO.start_date.replace('-', '')
                + '-' + PredictionDTO.end_date.replace('-', '')
                + ']-'
                + str(PredictionDTO.n_steps) + '-steps-'
                + str(int(PredictionDTO.test_size * 100)) + '%-test-'
                + str(PredictionDTO.univariate_epochs) + '-epochs-'
                + str(PredictionDTO.batch_size) + '-batches-'
                + str(int(PredictionDTO.dropout_rate * 100)) + '%-dropout'
        )
        return savedModelName


class Runner:

    @staticmethod
    def run_for_univariate_series_ir_spiltted_price(dataset, PredictionDTO, price, label, metrics):
        scaler = MinMaxScaler(feature_range=(0, 1))
        with_out_n_steps_point = len(dataset) - int(Config.n_steps)
        actuals = [round(data, 2) for data in dataset[price].tolist()][:with_out_n_steps_point]

        dates = dataset.index[:with_out_n_steps_point].tolist()
        dataset = dataset[price].values.reshape(-1, 1)
        dataset = scaler.fit_transform(dataset)

        end = pd.to_datetime(PredictionDTO.end_date)
        future_dates = pd.date_range(start=end, end=end + pd.Timedelta(days=PredictionDTO.n_predict_future_days),
                                     freq='W').tolist()

        results = {'labels': dates + future_dates, 'datasets': {}, 'actuals': actuals}

        for model in Config.models_name:
            if model in PredictionDTO.models:
                # print(f'[DEBUG] - in univariate of {model}')
                actuals, predictions, test_metrics, future_prediction = Univariate.splitted_univariate_series(model,
                                                                                                              dataset,
                                                                                                              scaler,
                                                                                                              label,
                                                                                                              PredictionDTO)
                results['datasets']['U-' + model + '-Predict'] = predictions + future_prediction
                if PredictionDTO.metrics is not None and len(PredictionDTO.metrics) > 0:
                    for metric in Config.metrics_name:
                        if metric in PredictionDTO.metrics:
                            metricLabel = label + '-' + metric
                            if not metrics.get(metricLabel, {}):
                                metrics[metricLabel] = {'labels': [], 'dataset': []}
                            if 'U-' + model not in metrics[metricLabel]['labels']:
                                metrics[metricLabel]['dataset'].append(test_metrics[metric])
                                metrics[metricLabel]['labels'].append('U-' + model)

        return results, metrics


if __name__ == '__main__':
    DTO = PredictionDTO()

    datasets = DataLoader.get_datasets_refactored(DTO)

    results, metrics = Runner.run_for_univariate_series_ir_spiltted_price(datasets[DTO.datasets[0]],
                                                                          DTO,
                                                                          DTO.prices[0],
                                                                          'U-CNN',
                                                                          {}
                                                                          )

    # labels is dates -> y
    print('\n [ + ] labels : ', results['labels'])
    # prediction values -> x
    print('\n [ + ] CNN model predictions : ', results['datasets']['U-CNN-Predict'])
    print('\n [ + ] LSTM model predictions : ', results['datasets']['U-LSTM-Predict'])
    # actual values -> x
    print('\n [ + ] actual values : ', results['actuals'])
    print('\n')
    print('\n [ + ] CNN model MAE : ', metrics['U-CNN-MAE']['dataset'][0])
    print('\n [ + ] LSTM model MAE : ', metrics['U-CNN-MAE']['dataset'][1])
