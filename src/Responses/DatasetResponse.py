from src.Config.Config import Config


class DatasetResponse:
    @staticmethod
    def total_response(datasets, requested_datasets, requested_prices):
        response = {}

        for title, dataset in datasets.items():
            if title in requested_datasets:
                response[title] = {'labels': [], 'datasets': {}}

                for price in Config.prices_name:
                    if price in requested_prices:
                        response[title]['labels'] = dataset.index.to_list()
                        response[title]['datasets'][price] = dataset[price].to_list()

        return response

