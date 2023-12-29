import numpy as np

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
    def run_for_univariate_series_ir_spiltted(dataset, models, prices):

        # data_mapping = {
        #     date: {Config.Low: low, Config.High: high, Config.Open: open, Config.Close: close}
        #     for date, low, high, open, close in
        #     zip(dataset.index, dataset[Config.Low].values.reshape(-1, 1), dataset[Config.High].values.reshape(-1, 1),
        #         dataset[Config.Open].values.reshape(-1, 1), dataset[Config.Close].values.reshape(-1, 1))
        # }

        # dynmiced
        data_mapping = {
            date: {col: value for col, value in zip(prices, row)}
            for date, *row in zip(dataset.index, *map(lambda col: dataset[col].values.reshape(-1, 1), prices))
        }
        # print(len(data_mapping))  # 287

        # dates = dataset.index.tolist()
        # dataset = dataset[Config.prediction_col].values.reshape(-1, 1)
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))

        results = {}
        for model in models:
            results[model] = {}
            for price in prices:
                dataset = [entry[price] for entry in data_mapping.values()]
                dataset = scaler.fit_transform(dataset)
                dates = list(data_mapping.keys())[:len(dataset) - int(Config.n_steps)]
                results[model][price] = Univariate.splitted_univariate_series(model, dataset, scaler, dates)

        return results

    @staticmethod
    def run_for_univariate_series_ir_spiltted_price(dataset, models, price):
        scaler = MinMaxScaler(feature_range=(0, 1))

        dates = dataset.index[:len(dataset) - int(Config.n_steps)]
        dataset = dataset[price].values.reshape(-1, 1)
        dataset = scaler.fit_transform(dataset)

        results = {'labels': list(dates), 'datasets': {}}
        for model in models:
            # print(f'[DEBUG] - in univariate of {model}')
            actuals, predictions = Univariate.splitted_univariate_series(model, dataset, scaler, dates)
            # actual = {
            #     index: {"date": data["date"], "actual": data["actual"]}
            #     for index, data in data_mapping.items()
            # }
            # predict = {
            #     index: {"date": data["date"], "predict": data["predict"]}
            #     for index, data in data_mapping.items()
            # }
            results['datasets']['U-' + model + '-Actual'] = actuals
            results['datasets']['U-' + model + '-Predict'] = predictions

        return results

    @staticmethod
    def run_for_multivariate_series_ir_spiltted(datasets, models, price, results, titles):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))

        stackedDataset, scaler = DataLoader.stack_datasets_splitted(datasets, price, scaler)

        dates = datasets[Config.Dollar].index[:len(stackedDataset) - int(Config.n_steps)].tolist()
        datasetTitles = list(datasets.keys())

        for model in models:
            # print(f'[DEBUG] - in multivariate of {model}')
            run = Multivariate.splitted_multivariate_series(model, stackedDataset, scaler, dates, datasetTitles)
            for title in titles:
                label = title + '-' + price

                if not results.get(label, {}):
                    results[label] = {'labels': list(dates), 'datasets': {}}

                results[label]['datasets']['M-' + model + '-Actual'] = run[title]['actual']
                results[label]['datasets']['M-' + model + '-Predict'] = run[title]['predict']

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
