import numpy as np
from flask import jsonify
from itertools import combinations


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

    @staticmethod
    def str_remove_flags(string):
        return string.replace('<', '').replace('>', '')

    @staticmethod
    def find_min_max_indexes(arr, k):
        # Check if the array has enough elements
        if len(arr) < int(k):
            k = len(arr)

        # Initialize lists to store the indexes of the minimum and maximum numbers
        min_indexes = []
        max_indexes = []

        # Iterate through the array
        for i, num in enumerate(arr):
            # Update min_indexes based on the current element
            if len(min_indexes) < k:
                min_indexes.append(i)
            else:
                # Find the index of the maximum element in min_indexes
                max_index = min_indexes.index(max(min_indexes, key=lambda x: arr[x]))

                # Replace the largest element's index if the current element is smaller
                if num < arr[min_indexes[max_index]]:
                    min_indexes[max_index] = i

            # Update max_indexes based on the current element
            if len(max_indexes) < k:
                max_indexes.append(i)
            else:
                # Find the index of the minimum element in max_indexes
                min_index = max_indexes.index(min(max_indexes, key=lambda x: arr[x]))

                # Replace the smallest element's index if the current element is larger
                if num > arr[max_indexes[min_index]]:
                    max_indexes[min_index] = i

        # Return both the indexes of the minimum and maximum numbers
        return min_indexes, max_indexes

    @staticmethod
    def extract_combinations(input_array):
        result = []

        # Generate combinations of different lengths
        for r in range(2, len(input_array) + 1):
            # Get combinations of length r
            for combo in combinations(input_array, r):
                # Check if the combination contains unique and non-singleton elements
                if len(set(combo)) == len(combo) and len(combo) > 1:
                    result.append(list(combo))

        return result

    @staticmethod
    def flatten_list(nested_list):
        flat_list = []
        for sublist in nested_list:
            if isinstance(sublist, list):
                flat_list.extend(Helper.flatten_list(sublist))
            else:
                flat_list.append(sublist)
        return flat_list
