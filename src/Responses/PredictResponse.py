import numpy as np

from src.Config.Config import Config
from src.Helper.Helper import Helper
from src.Runner import Runner


class PredictResponse:
    @staticmethod
    def detailed_response(datasets, requested_datasets, requested_models, requested_prices, requested_series):
        if Config.multivariate in requested_series:
            multivariates = Runner.run_for_multivariate_series_ir(datasets)
            # multivariates = Runner.run_for_multivariate_series_ir_spiltted(datasets, requested_models)
        results = {}
        for title, dataset in datasets.items():
            results[title] = {}
            if title in requested_datasets:
                if Config.univariate in requested_series:
                    results[title][Config.univariate] = Runner.run_for_univariate_series_ir_spiltted(dataset,
                                                                                                     requested_models,
                                                                                                     requested_prices)
                else:
                    results[title][Config.univariate] = None
                if Config.multivariate in requested_series:
                    results[title][Config.multivariate] = multivariates[title]
                else:
                    results[title][Config.multivariate] = None
            else:
                results[title][Config.univariate] = None
                results[title][Config.multivariate] = None
        return results

    @staticmethod
    def total_response(datasets, requested_datasets, requested_models, requested_prices, requested_series,
                       requested_metrics):
        results = {}
        metrics = {}

        for title, dataset in datasets.items():
            for price in Config.prices_name:
                if price in requested_prices and title in requested_datasets:
                    label = title + '-' + price
                    # print(f'[DEBUG] - label is for {price} : ', label)

                    if Config.univariate in requested_series:
                        # print(f'[DEBUG] - in univariate for {label}')
                        results[label], metrics = Runner.run_for_univariate_series_ir_spiltted_price(dataset,
                                                                                                     requested_models,
                                                                                                     price,
                                                                                                     requested_metrics,
                                                                                                     label, metrics)
                    if Config.multivariate in requested_series:
                        # print(f'[DEBUG] - in multivariate for {label}')
                        results, metrics = Runner.run_for_multivariate_series_ir_spiltted(datasets, requested_models,
                                                                                          price,
                                                                                          results, requested_datasets,
                                                                                          requested_metrics, metrics)

        return results, metrics

    @staticmethod
    def add_ensemble_models_to_response(results, metrics, n_top_models_to_ensemble, apply_combinations=False):
        ensemble = None
        top_models = []
        for label, data in metrics.items():
            # for ensemble need to have at least 2 models
            if len(data['dataset']) < 2:
                break
            min_indexes, max_indexes = Helper.find_min_max_indexes(data['dataset'], n_top_models_to_ensemble)
            if Config.MAE in label or Config.MAPE in label or Config.MSE in label or Config.RMSE in label:
                top_models = [data['labels'][index] for index in min_indexes]
            elif Config.R2 in label:
                top_models = [data['labels'][index] for index in max_indexes]
            break

        if len(top_models) >= 2:
            if apply_combinations:
                combinations = Helper.extract_combinations(top_models)
                for top_models_comb in combinations:
                    PredictResponse.add_ensemble_to_response(ensemble, results, top_models_comb)
            else:
                PredictResponse.add_ensemble_to_response(ensemble, results, top_models)

        return results, metrics

    @staticmethod
    def add_ensemble_to_response(ensemble, results, top_models):
        for label, data in results.items():
            for model, prediction in data['datasets'].items():
                # for example convert 'M-CNN-Predict' to 'M-CNN'
                model = '-'.join(model.split('-')[:-1])
                if model in top_models:
                    # train_point = round(len(data['actuals']) - (len(data['actuals']) * Config.test_size))
                    # test_predictions = prediction[train_point:]
                    if ensemble is None:
                        ensemble = np.array(prediction)
                    else:
                        ensemble += np.array(prediction)
            if ensemble is not None:
                ensemble /= len(top_models)
                if results.get(label, {}):
                    ensemble_title = '-'.join(top_models) + '-(Ensembled)-Predict'
                    results[label]['datasets'][ensemble_title] = ensemble.tolist()
