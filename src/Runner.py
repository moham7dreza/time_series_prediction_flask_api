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
    def run_for_univariate_series_ir_spiltted_price(dataset, models, price, metrics, label, metricsResult):
        scaler = MinMaxScaler(feature_range=(0, 1))

        actuals = [round(data, 2) for data in dataset[price].tolist()]

        dates = dataset.index[:len(dataset) - int(Config.n_steps)]
        dataset = dataset[price].values.reshape(-1, 1)
        dataset = scaler.fit_transform(dataset)

        results = {'labels': list(dates), 'datasets': {}, 'actuals': actuals}

        for model in Config.models_name:
            if model in models:
                # print(f'[DEBUG] - in univariate of {model}')
                actuals, predictions, test_metrics = Univariate.splitted_univariate_series(model,
                                                                                           dataset,
                                                                                           scaler, dates)
                # print('metrics after run univariate : ', Evaluation.calculateMetrics(np.array(actuals), np.array(predictions)))
                # actual = {
                #     index: {"date": data["date"], "actual": data["actual"]}
                #     for index, data in data_mapping.items()
                # }
                # predict = {
                #     index: {"date": data["date"], "predict": data["predict"]}
                #     for index, data in data_mapping.items()
                # }
                # results['datasets']['U-' + model + '-Actual'] = actuals TODO actuals removed
                results['datasets']['U-' + model + '-Predict'] = predictions
                for metric in Config.metrics_name:
                    if metric in metrics:
                        metricLabel = label + '-' + metric
                        if not metricsResult.get(metricLabel, {}):
                            metricsResult[metricLabel] = {'labels': [], 'dataset': []}
                        if 'U-' + model not in metricsResult[metricLabel]['labels']:
                            metricsResult[metricLabel]['dataset'].append(test_metrics[metric])
                            metricsResult[metricLabel]['labels'].append('U-' + model)
        return results, metricsResult

    @staticmethod
    def run_for_multivariate_series_ir_spiltted(datasets, models, price, results, titles, requested_metrics, metrics):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        print('[DEBUG] price : ', price)
        stackedDataset, scaler = DataLoader.stack_datasets_splitted(datasets, price, scaler)

        dates = datasets[Config.Dollar].index[:len(stackedDataset) - int(Config.n_steps)].tolist()
        datasetTitles = list(datasets.keys())

        for model in Config.models_name:
            if model in models:
                # print(f'[DEBUG] - in multivariate of {model}')
                run, test_metrics = Multivariate.splitted_multivariate_series(model, stackedDataset, scaler, dates,
                                                                              datasetTitles)
                print('[DEBUG] - model : ', model)

                for title in titles:
                    label = title + '-' + price
                    print('[DEBUG] - label : ', label)
                    actuals = [round(data, 2) for data in datasets[title][price].tolist()]

                    if not results.get(label, {}):
                        results[label] = {'labels': list(dates), 'datasets': {}, 'actuals': actuals}

                    # results[label]['datasets']['M-' + model + '-Actual'] = run[title]['actual'] TODO actuals removed
                    results[label]['datasets']['M-' + model + '-Predict'] = run[title]['predict']
                    # results[label]['metrics']['M-' + model] = {
                    #     'MAE': run[title]['mae'],
                    #     'MAPE': run[title]['mape'],
                    #     'MSE': run[title]['mse'],
                    #     'R2': run[title]['r2'],
                    #     'RMSE': run[title]['rmse']
                    # }

                    for metric in Config.metrics_name:
                        if metric in requested_metrics:
                            print('[DEBUG] - metrics before : ', metrics)
                            metricLabel = label + '-' + metric
                            print('[DEBUG] - metric label', metricLabel)
                            if not metrics.get(metricLabel, {}):
                                print('[DEBUG] - metric label not found')
                                metrics[metricLabel] = {'labels': [], 'dataset': []}
                            if 'M-' + model not in metrics[metricLabel]['labels']:
                                metrics[metricLabel]['labels'].append('M-' + model)
                                print(f'[DEBUG] - add M-{model} to metrics[{metricLabel}][labels]')
                                print(f'[DEBUG] - metrics[{metricLabel}][labels] is : ', metrics[metricLabel]['labels'])
                                metrics[metricLabel]['dataset'].append(test_metrics[title][metric])
                                print(f'[DEBUG] - add {test_metrics[title][metric]} to metrics[{metricLabel}][dataset]')
                                print(f'[DEBUG] - metrics[{metricLabel}][dataset] is : ', metrics[metricLabel]['dataset'])

        return results, metrics

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
