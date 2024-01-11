from src.Config.Config import Config


class DatasetResponse:
    @staticmethod
    def total_response(datasets, PredictionDTO):
        response = {}

        for title, dataset in datasets.items():
            if title in PredictionDTO.datasets:
                response[title] = {'labels': [], 'datasets': {}}

                for price in Config.prices_name:
                    if price in PredictionDTO.prices:
                        response[title]['labels'] = dataset.index.to_list()
                        response[title]['datasets'][price] = dataset[price].to_list()

        return response
