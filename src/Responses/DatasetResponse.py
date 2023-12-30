from src.Config.Config import Config


class DatasetResponse:
    @staticmethod
    def total_response(datasets, requested_datasets, requested_prices):
        results = {'labels': None, 'datasets': {}}
        response = {}
        for title, dataset in datasets.items():
            for price in Config.prices_name:
                if price in requested_prices and title in requested_datasets:
                    results['labels'] = dataset.index.to_list()
                    results['datasets'][price] = dataset[price].to_list()
                    response[title] = results

        return response
