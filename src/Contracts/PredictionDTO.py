from flask import request


class PredictionDTO:
    def __init__(self):
        requests = request.get_json() or {}
        self.n_steps = requests.get("n_steps")
        self.datasets = requests.get("dataset")
        self.models = requests.get("model")
        self.series = requests.get("serie")
        self.prices = requests.get("price")
        self.metrics = requests.get("metric")
        self.n_top_models_to_ensemble = requests.get("n_top_models_to_ensemble")
        self.apply_combinations = requests.get("apply_combinations")
        self.n_predict_future_days = requests.get("n_predict_future_days")
        self.start_date = requests.get("start_date")
        self.end_date = requests.get("end_date")
