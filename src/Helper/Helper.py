from numpy import array


class Helper:
    @staticmethod
    def flatten_arr(result):
        items = array(result).flatten()
        return [round(item, 2) for item in items]
