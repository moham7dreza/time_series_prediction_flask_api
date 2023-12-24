from src.Config.Config import Config
from src.Data.DataLoader import DataLoader
from src.Series.Multivariate import Multivariate
from src.Series.Univariate import Univariate
from sklearn.preprocessing import MinMaxScaler


class Runner:
    @staticmethod
    def run_for_univariate_series_ir(dataset):
        dataset = dataset[Config.prediction_col].values

        train, test, u_last = DataLoader.train_test_split(dataset)

        u_cnn = Univariate.univariant_series(Config.CNN, train, test)
        u_lstm = Univariate.univariant_series(Config.LSTM, train, test)
        u_b_lstm = Univariate.univariant_series(Config.bi_LSTM, train, test)
        u_gru = Univariate.univariant_series(Config.GRU, train, test)
        u_b_gru = Univariate.univariant_series(Config.bi_GRU, train, test)
        u_ann = Univariate.univariant_series(Config.ANN, train, test)
        u_b_ann = Univariate.univariant_series(Config.bi_ANN, train, test)
        u_rnn = Univariate.univariant_series(Config.RNN, train, test)
        u_b_rnn = Univariate.univariant_series(Config.bi_RNN, train, test)

        return {
            'U-' + Config.CNN: u_cnn[0],
            'U-' + Config.LSTM: u_lstm[0],
            'U-' + Config.bi_LSTM: u_b_lstm[0],
            'U-' + Config.GRU: u_gru[0],
            'U-' + Config.bi_GRU: u_b_gru[0],
            'U-' + Config.ANN: u_ann[0],
            'U-' + Config.bi_ANN: u_b_ann[0],
            'U-' + Config.RNN: u_rnn[0],
            'U-' + Config.bi_RNN: u_b_rnn[0],
            'REAL': u_last[0]
        }

    @staticmethod
    def run_for_univariate_series_ir_spiltted(dataset, models):
        dates = dataset.index.tolist()
        dataset = dataset[Config.prediction_col].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        results = {}
        for model in models:
            results[model] = Univariate.splitted_univariate_series(model, dataset, scaler, dates)

        return results

    @staticmethod
    def run_for_multivariate_series_ir_spiltted(datasets, models):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))

        dataset, scaler = DataLoader.stack_datasets_splitted(datasets, scaler)

        dates = datasets[Config.Dollar].index.tolist()

        titles = list(datasets.keys())
        results = {}
        for i in range(len(titles)):
            results[titles[i]] = {}
            for model in models:
                results[titles[i]][model] = Multivariate.splitted_multivariate_series(model, dataset, scaler, dates)

        return results

    @staticmethod
    def run_for_multivariate_series_ir(datasets):
        dataset = DataLoader.stack_datasets(datasets)

        train, test, last = DataLoader.train_test_split(dataset)

        cnn = Multivariate.multivariate_multiple_output_parallel_series(Config.CNN, train, test)
        lstm = Multivariate.multivariate_multiple_output_parallel_series(Config.LSTM, train, test)
        b_lstm = Multivariate.multivariate_multiple_output_parallel_series(Config.bi_LSTM, train, test)
        gru = Multivariate.multivariate_multiple_output_parallel_series(Config.GRU, train, test)
        b_gru = Multivariate.multivariate_multiple_output_parallel_series(Config.bi_GRU, train, test)
        ann = Multivariate.multivariate_multiple_output_parallel_series(Config.ANN, train, test)
        b_ann = Multivariate.multivariate_multiple_output_parallel_series(Config.bi_ANN, train, test)
        rnn = Multivariate.multivariate_multiple_output_parallel_series(Config.RNN, train, test)
        b_rnn = Multivariate.multivariate_multiple_output_parallel_series(Config.bi_RNN, train, test)

        titles = list(datasets.keys())
        results = {}
        for i in range(len(titles)):
            results[titles[i]] = {
                'M-' + Config.CNN: cnn[i],
                'M-' + Config.LSTM: lstm[i],
                'M-' + Config.bi_LSTM: b_lstm[i],
                'M-' + Config.GRU: gru[i],
                'M-' + Config.bi_GRU: b_gru[i],
                'M-' + Config.ANN: ann[i],
                'M-' + Config.bi_ANN: b_ann[i],
                'M-' + Config.RNN: rnn[i],
                'M-' + Config.bi_RNN: b_rnn[i],
                'REAL': last[i]
            }

        return results
