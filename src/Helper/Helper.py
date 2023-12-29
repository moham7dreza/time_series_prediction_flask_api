import numpy as np
from flask import jsonify


class Helper:
    @staticmethod
    def flatten_arr(result):
        items = np.array(result).flatten()
        return [round(item, 2) for item in items]

    # Convert NumPy float32 to Python float
    @staticmethod
    def convert_to_python_float(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [Helper.convert_to_python_float(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: Helper.convert_to_python_float(value) for key, value in obj.items()}
        return obj

    @staticmethod
    def merge_and_clean(round_decimals=None, **arrays):
        # Round each array if round_decimals is specified
        rounded_arrays = [np.round(arr, decimals=round_decimals) if round_decimals is not None else arr for arr in
                          arrays.values()]

        # Concatenate the rounded arrays
        concatenated_array = np.concatenate(rounded_arrays).ravel()

        # Convert to a Python list
        return concatenated_array.tolist()

    @staticmethod
    def failed_response(method, msg):
        if method != 'POST':
            return jsonify(
                {
                    'status': 'ok',
                    'message': 'invalid request method'
                }
            ), 400

    @staticmethod
    def implode(array):
        return ', '.join([str(item) for item in array])
