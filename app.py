from flask import Flask, jsonify, request
from flask_cors import CORS
from src.Config.Config import Config
from src.Data.DataLoader import DataLoader
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

    # Config.setNSteps(requests.get('n_steps'))
    # requested_datasets = requests.get('datasets')
    requested_datasets = [Config.Dollar]
    requested_models = requests.get('models')
    requested_series = requests.get('series')

    datasets = DataLoader.get_datasets()
    results = {}
    for dataset_name in requested_datasets:
        # print(dataset_name)
        univariates = Runner.run_for_univariate_series_ir(datasets[dataset_name])
        results[dataset_name] = univariates

    return jsonify(
        {
            'status': 'ok',
            'data': results
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
