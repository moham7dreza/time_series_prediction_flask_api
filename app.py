import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.Config.Config import Config
from src.Data.DataLoader import DataLoader
from src.Helper.Helper import Helper
from src.Runner import Runner

app = Flask(__name__)
CORS(app)


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
    requests = request.get_json()

    Config.setNSteps(requests.get('n_steps'))
    requested_datasets = requests.get('dataset')
    requested_models = requests.get('model')
    requested_series = requests.get('serie')
    requested_prices = requests.get('price')

    datasets = DataLoader.get_datasets()

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

    return jsonify(
        {
            'status': 'ok',
            'data': Helper.convert_to_python_float(results)
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
