from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score, r2_score
import numpy as np


class Evaluation:
    @staticmethod
    def calculateMetricsUnivariate(actuals, predictions):
        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(actuals, predictions)
        # print("Mean Absolute Error (MAE):", mae)

        # Mean Squared Error (MSE)
        mse = mean_squared_error(actuals, predictions)
        # print("Mean Squared Error (MSE):", mse)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        # print("Root Mean Squared Error (RMSE):", rmse)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        # print("Mean Absolute Percentage Error (MAPE):", mape)

        # R-squared (R2)
        r2 = r2_score(actuals, predictions)
        # print("R-squared (R2):", r2)

        return {
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'R2': round(r2, 2),
        }
