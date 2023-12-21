from flask import Flask, jsonify, request
from flask_cors import CORS
import os

from src.Config.Config import Config
from src.Data.DataLoader import DataLoader

app = Flask(__name__)
CORS(app)


@app.route('/')
def attack():
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


if __name__ == '__main__':
    app.run(debug=True)
