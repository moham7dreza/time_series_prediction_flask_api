from src.Config.Config import Config
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
    def total_response(datasets, requested_datasets, requested_models, requested_prices, requested_series):
        results = {}

        for title, dataset in datasets.items():
            for price in Config.prices_name:
                if price in requested_prices and title in requested_datasets:
                    label = title + '-' + price
                    # print(f'[DEBUG] - label is for {price} : ', label)

                    if Config.univariate in requested_series:
                        # print(f'[DEBUG] - in univariate for {label}')
                        results[label] = Runner.run_for_univariate_series_ir_spiltted_price(dataset,
                                                                                            requested_models,
                                                                                            price)
                    if Config.multivariate in requested_series:
                        # print(f'[DEBUG] - in multivariate for {label}')
                        results = Runner.run_for_multivariate_series_ir_spiltted(datasets, requested_models, price,
                                                                                 results, requested_datasets)

        return results
