import numpy as np

from src.Config.Config import Config
from src.Helper.Helper import Helper
from src.Model.Evaluation import Evaluation
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
                    PredictResponse.calc_top_models_mean(ensemble, results, top_models_comb, metrics)
            else:
                PredictResponse.calc_top_models_mean(ensemble, results, top_models, metrics)

        return results, metrics

    @staticmethod
    def calc_top_models_mean(ensemble, results, top_models, metrics):
        for label, data in results.items():
            actuals = data['actuals']
            train_point = round(len(actuals) - (len(actuals) * Config.test_size))
            test_actuals = actuals[train_point:]

            for model, prediction in data['datasets'].items():
                # for example convert 'M-CNN-Predict' to 'M-CNN'
                model = '-'.join(model.split('-')[:-1])
                if model in top_models:
                    if ensemble is None:
                        ensemble = np.array(prediction)
                    else:
                        ensemble += np.array(prediction)
            if ensemble is not None:
                ensemble /= len(top_models)

                PredictResponse.add_ensemble_metrics_to_response(ensemble, label, metrics, test_actuals, top_models,
                                                                 train_point)

                if results.get(label, {}):
                    ensemble_title = '-'.join(top_models) + '-(Ensembled)-Predict'
                    results[label]['datasets'][ensemble_title] = ensemble.tolist()

    @staticmethod
    def add_ensemble_metrics_to_response(ensemble, label, metrics, test_actuals, top_models, train_point):
        y_test = np.array(test_actuals)
        test_predictions = ensemble[train_point:]
        ensemble_metrics = Evaluation.calculateMetricsUnivariate(y_test, test_predictions)
        for metric_label, metric_data in metrics.items():
            for ensemble_metric_label, ensemble_metric_data in ensemble_metrics.items():
                # Dollar-<CLOSE> and MAE in Dollar-<CLOSE>-MAE
                if label in metric_label and ensemble_metric_label in metric_label:
                    if metrics.get(metric_label, {}):
                        metrics[metric_label]['dataset'].append(ensemble_metric_data)
                        metrics[metric_label]['labels'].append('-'.join(top_models))
                    break
