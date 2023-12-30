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

    @staticmethod
    def calculateMetricsMultivariate(actuals, predictions):
        actuals = np.array(actuals)
        actuals = [actuals[:, i].reshape((actuals.shape[0], 1)) for i in range(len(predictions))]
        # print(np.array(actuals), np.array(predictions))  # (5, 57, 1) (5, 57, 1)
        actuals = np.squeeze(actuals)
        predictions = np.squeeze(predictions)

        # Mean Absolute Error (MAE)
        mae = [round(mean_absolute_error(actuals[i], predictions[i]), 2) for i in range(len(actuals))]
        # print("Mean Absolute Error (MAE):", mae)

        # Mean Squared Error (MSE)
        mse = [round(mean_squared_error(actuals[i], predictions[i]), 2) for i in range(len(actuals))]
        # print("Mean Squared Error (MSE):", mse)

        # Root Mean Squared Error (RMSE)
        rmse = np.round(np.sqrt(np.array(mse)), 2).tolist()
        # print("Root Mean Squared Error (RMSE):", rmse)

        # Mean Absolute Percentage Error (MAPE)
        mape = [round(np.mean(np.abs((actuals[i] - predictions[i]) / actuals[i])) * 100, 2) for i in
                range(len(predictions))]
        # print("Mean Absolute Percentage Error (MAPE):", mape)

        # R-squared (R2)
        r2 = [round(r2_score(actuals[i], predictions[i]), 2) for i in range(len(actuals))]
        # print("R-squared (R2):", r2)

        # return {
        #     'MAE': mae,
        #     'MSE': mse,
        #     'RMSE': rmse,
        #     'MAPE': mape,
        #     'R2': r2,
        # }

        return mae, mse, rmse, mape, r2
