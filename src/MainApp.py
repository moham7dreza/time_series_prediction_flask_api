import pandas as pd

from pandora.Config.Config import Config
from pandora.Data.DataLoader import DataLoader
from pandora.Series.Multivariate import Multivariate
from pandora.Plot.ResultPlotter import ResultPlotter
from pandora.Runner import Runner
from pandora.Series.Univariate import Univariate

from google.colab import drive


class MainApp:
    def __init__(self):
        drive.mount('/content/drive')

    @staticmethod
    def run_and_plot(title, dataset, multivariates):
        # 10. run algorithms on univariate dollar
        univariates = Runner.run_for_univariate_series_ir(dataset)
        data = {**univariates, **multivariates}
        labels = list(data.keys())
        results = list(data.values())
        # 13. plot all results of algorithms on car
        ResultPlotter.plot_result(labels, results, title)

    @staticmethod
    def attack():
        # 3. get each dataset according to env
        if Config.colab:
            dfs = DataLoader.read_csv_files_from_drive_in_colab(Config.drive_csv_folder_path)

            ir_dollar = dfs[Config.dollar_file_name]
            ir_home = dfs[Config.home_file_name]
            ir_oil = dfs[Config.oil_file_name]
            ir_car = dfs[Config.car_file_name]
            ir_gold = dfs[Config.gold_file_name]
        else:
            ir_dollar = pd.read_csv(Config.local_csv_folder_path + Config.dollar_file_name)
            ir_home = pd.read_csv(Config.local_csv_folder_path + Config.home_file_name)
            ir_oil = pd.read_csv(Config.local_csv_folder_path + Config.oil_file_name)
            ir_car = pd.read_csv(Config.local_csv_folder_path + Config.car_file_name)
            ir_gold = pd.read_csv(Config.local_csv_folder_path + Config.gold_file_name)

        # 5. preprocess each dataset (sort, date col check, set index, resample)
        ir_dollar = DataLoader.data_preprocessing(ir_dollar, format=False)
        ir_home = DataLoader.data_preprocessing(ir_home)
        ir_oil = DataLoader.data_preprocessing(ir_oil)
        ir_car = DataLoader.data_preprocessing(ir_car)
        ir_gold = DataLoader.data_preprocessing(ir_gold)

        # 7. array datasets for horizontally stack in multivariate prediction
        datasets = {
            Config.Dollar: ir_dollar,
            Config.Home: ir_home,
            Config.Oil: ir_oil,
            Config.Car: ir_car,
            Config.Gold: ir_gold
        }

        # 6. plot each dataset
        if Config.plotting:
            for title, dataset in datasets.items():
                ResultPlotter.plotting(dataset, title)

        # 9. run algorithms on multivariate series
        multivariates = Runner.run_for_multivariate_series_ir(datasets)
        for title, dataset in datasets.items():
            MainApp.run_and_plot(title, dataset, multivariates[title])
