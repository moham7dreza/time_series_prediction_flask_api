from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_migrate import Migrate

from src.Config.Config import Config
from src.Config.app import Config as appConfig
from src.Data.DataLoader import DataLoader
from src.Helper.Helper import Helper
from src.Migrations import db
from src.Responses.DatasetResponse import DatasetResponse
from src.Responses.PredictResponse import PredictResponse
from src.Services import PredictService

app = Flask(__name__)
CORS(app)
app.config.from_object(appConfig)
db.init_app(app)
migrate = Migrate(app, db)


class PredictionDTO:
    def __init__(self):
        requests = request.get_json()

        self.n_steps = requests.get("n_steps")
        self.datasets = requests.get("dataset")
        self.models = requests.get("model")
        self.series = requests.get("serie")
        self.prices = requests.get("price")
        self.metrics = requests.get("metric")
        self.n_top_models_to_ensemble = requests.get("n_top_models_to_ensemble")
        self.apply_combinations = requests.get("apply_combinations")
        self.n_predict_future_days = requests.get("n_predict_future_days")
        self.start_date = requests.get("startDate")
        self.end_date = requests.get("endDate")
        self.epochs = requests.get("epochs")
        self.batch_size = requests.get("batch_size")
        try:
            self.test_size = requests.get("test_size") / 100
            self.dropout_rate = requests.get("dropout_rate") / 100
        except (TypeError, ValueError):
            self.test_size = None
            self.dropout_rate = None


@app.route('/')
def main():
    return jsonify(
        {
            'status': 'OK',
            'data': [pred.serialize() for pred in PredictService.index()]
        }
    ), 201


@app.route('/last-predict-props')
def last_record():
    return jsonify(
        {
            'status': 'OK',
            'data': PredictService.latest().serialize()
        }
    ), 201


@app.route('/datasets', methods=['POST'])
def get_datasets():
    DTO = PredictionDTO()

    datasets = DataLoader.get_datasets_refactored(DTO)

    results = DatasetResponse.total_response(datasets, DTO)

    return jsonify(
        {
            'status': 'OK',
            'data': results
        }
    )


@app.route('/models-name')
def get_models_name():
    return jsonify(
        {
            'status': 'OK',
            'data': Config.models_name
        }
    )


@app.route('/metrics-name')
def get_metrics_name():
    return jsonify(
        {
            'status': 'OK',
            'data': Config.metrics_name
        }
    )


@app.route('/prices-name')
def get_prices_name():
    return jsonify(
        {
            'status': 'OK',
            'data': Config.prices_name
        }
    )


@app.route('/datasets-name')
def get_datasets_name():
    return jsonify(
        {
            'status': 'OK',
            'data': Config.datasets_name
        }
    )


@app.route('/series-name')
def get_series_name():
    return jsonify(
        {
            'status': 'OK',
            'data': Config.series_name
        }
    )


@app.route('/make-prediction', methods=['POST'])
def make_prediction():
    if request.method != 'POST':
        return jsonify(
            {
                'status': 'ok',
                'message': 'invalid request method'
            }
        ), 400

    requests = request.get_json()

    DTO = PredictionDTO()

    PredictService.create(requests)

    datasets = DataLoader.get_datasets_refactored(DTO)

    results, metrics = PredictResponse.total_response(datasets, DTO)

    if DTO.n_top_models_to_ensemble > 0:
        results, metrics = PredictResponse.add_ensemble_models_to_response(results, metrics, DTO)

    return jsonify(
        {
            'status': 'ok',
            'data': Helper.convert_to_python_float(results),
            'metrics': Helper.convert_to_python_float(metrics)
        }
    )


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
