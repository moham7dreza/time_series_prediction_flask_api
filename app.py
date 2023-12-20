from flask import Flask, jsonify, request
from flask_cors import CORS
import os

from src.Data.DataLoader import DataLoader

app = Flask(__name__)
CORS(app)


@app.route('/')
def attack():
    datasets = DataLoader.get_datasets()
    return jsonify(list(datasets.keys()))


if __name__ == '__main__':
    app.run(debug=True)
