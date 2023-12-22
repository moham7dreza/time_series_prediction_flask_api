import numpy as np


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
