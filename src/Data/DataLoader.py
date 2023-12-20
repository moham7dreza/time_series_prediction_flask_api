import io
import os

import pandas as pd
from google.colab import files
from numpy import hstack

from pandora.Config.Config import Config
from pandora.Helper.Helper import Helper


class DataLoader:
    @staticmethod
    def upload_and_read_csv_in_colab(name):
        print('\nSelect file for ' + name + ' ...\n')
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        if name.lower() not in file_name.lower():
            raise ValueError("uploaded dataset is incorrect")
        file_content = uploaded[file_name]
        print('\nSelected file ' + file_name + ' is begin to read ...\n')
        # load dataset
        series = pd.read_csv(io.BytesIO(file_content))
        return series

    @staticmethod
    def read_csv_files_from_drive_in_colab(folder_path=Config.drive_csv_folder_path):
        # Navigate to the folder containing your CSV files
        %cd $folder_path

        # Get a list of all CSV files in the folder
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        # Read each CSV file into a DataFrame and store them in a dictionary
        dataframes = {}
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            dataframes[file] = pd.read_csv(file_path)
        return dataframes

    @staticmethod
    def stack_datasets(datasets, col=Config.prediction_col):
        sequences = []
        for dataset in list(datasets.values()):
            seq = dataset[col].values
            sequences.append(seq)
            # print(len(seq))
        return hstack([seq.reshape(len(seq), 1) for seq in sequences])

    @staticmethod
    def data_preprocessing(dataset, date_col=Config.date_col, start_date=Config.start_date, end_date=Config.end_date,
                           format=True):
        # sort by date
        dataset = dataset.sort_values(by=date_col)
        # print(dataset)

        if (format):
            # Assuming 'date' is the column containing date in integer format
            dataset[date_col] = pd.to_datetime(dataset[date_col], format='%Y%m%d')
        else:
            dataset[date_col] = pd.to_datetime(dataset[date_col])

        # Convert object columns to strings
        object_columns = dataset.select_dtypes(include='object').columns
        dataset[object_columns] = dataset[object_columns].astype(str)

        # print(dataset.dtypes)

        # Identify and exclude object columns
        non_object_columns = dataset.select_dtypes(exclude='object').columns
        # Create a new DataFrame without object columns
        dataset = dataset[non_object_columns]

        # print(dataset)
        dataset = dataset.set_index(date_col)
        # print(dataset.dtypes)

        dataset = dataset.resample('W-Sat').mean().ffill()
        # print(dataset)

        dataset = dataset.loc[start_date:end_date]
        # print(dataset)

        return dataset

    @staticmethod
    def train_test_split(dataset, test_size=Config.n_steps):
        index = -test_size - 1
        train = dataset[:index]
        test = dataset[index:-1]
        last = Helper.flatten_arr(dataset[-1])

        return train, test, last
