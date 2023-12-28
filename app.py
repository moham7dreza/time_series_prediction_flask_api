from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from src.Config.Config import Config
from src.Data.DataLoader import DataLoader
from src.Helper.Helper import Helper
from src.Migrations import db
from src.Responses.PredictResponse import PredictResponse
from src.Config.app import Config as appConfig
from src.Services import PredictService

app = Flask(__name__)
CORS(app)
app.config.from_object(appConfig)
db.init_app(app)


@app.route('/datasets')
def get_datasets():
    datasets = DataLoader.get_datasets()
    response = {}
    for name, dataset in datasets.items():
        response[name] = {index + 1: {"date": date, "close": close} for index, (date, close) in
                          enumerate(zip(dataset.index.to_list(), dataset[Config.prediction_col].to_list()))}

    return jsonify(
        {
            'status': 'OK',
            'data': response
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
    print(Predict.query.all())
    requests = request.get_json()

    Config.setNSteps(requests.get('n_steps'))
    requested_datasets = requests.get('dataset')
    requested_models = requests.get('model')
    requested_series = requests.get('serie')
    requested_prices = requests.get('price')

    datasets = DataLoader.get_datasets()

    results = PredictResponse.total_response(datasets, requested_datasets, requested_models, requested_prices,
                                             requested_series)

    return jsonify(
        {
            'status': 'ok',
            'data': Helper.convert_to_python_float(results)
        }
    )


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
