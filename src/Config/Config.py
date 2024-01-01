import os


class Config:
    colab = False
    plotting = False
    date_col = "<DTYYYYMMDD>"
    # start_date = '2017-06-10'
    start_date = '2014-01-07'
    # end_date = '2022-12-03'
    end_date = '2022-10-10'
    n_steps = 3
    epochs_for_multivariate_series = 100
    epochs_for_univariate_series = 100
    dropout_rate = 0.2
    prediction_col = '<CLOSE>'
    local_csv_folder_path = './iran_stock'
    drive_csv_folder_path = '/content/drive/My Drive/iran_stock'
    drive_model_folder_path = '/content/drive/My Drive/time_series_models'
    local_model_folder_path = './time_series_models'
    # 1. set csv dataset file names
    dollar_file_name = 'dollar_tjgu_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    am_car_file_name = 'Iran.Kh..A..M._from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    housing_file_name = 'Housing.Inv.from_2004.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'
    kh_gold_file_name = 'Kharazmi.Info._GOLD_from_2014.csv'
    # dataset titles
    titles = ['Dollar', 'Home', 'Oil', 'Car', 'Gold', ]
    Dollar = 'Dollar'
    Home = 'Home'
    Oil = 'Oil'
    Car = 'Car'
    Gold = 'Gold'
    datasets_name = [Dollar, Home, Oil, Car, Gold]
    # model names
    CNN = 'CNN'
    LSTM = 'LSTM'
    bi_LSTM = 'B_LSTM'
    GRU = 'GRU'
    bi_GRU = 'B_GRU'
    ANN = 'ANN'
    bi_ANN = 'B_ANN'
    RNN = 'RNN'
    bi_RNN = 'B_RNN'
    RF_REGRESSOR = 'RF_Regressor'
    GB_REGRESSOR = 'GB_Regressor'
    DT_REGRESSOR = 'DT_Regressor'
    XGB_REGRESSOR = 'XGB_Regressor'
    Linear_REGRESSION = 'Linear_REGRESSION'
    models_name = [
        CNN,
        LSTM,
        bi_LSTM,
        GRU,
        bi_GRU,
        ANN,
        bi_ANN,
        RNN,
        bi_RNN,
        # RF_REGRESSOR,
        # GB_REGRESSOR,
        # DT_REGRESSOR,
        # XGB_REGRESSOR,
        # Linear_REGRESSION,
    ]
    # series type
    univariate = 'univariate'
    multivariate = 'multivariate'
    series_name = [univariate, multivariate]
    base_project_path = os.path.abspath(os.path.dirname(__file__))
    plot_labels = ['M-CNN', 'U-CNN',
                   'M-B-LSTM', 'U-B-LSTM', 'M-LSTM', 'U-LSTM',
                   'M-GRU', 'U-GRU', 'M-B-GRU', 'U-B-GRU',
                   'M-ANN', 'U-ANN', 'M-B-ANN', 'U-N-ANN',
                   'M-RNN', 'U-RNN', 'M-B-RNN', 'U-N-RNN',
                   'REAL']
    checkForModelExistsInFolder = True
    test_size = 0.2
    random_state = 42
    estimators = 100
    learning_rate = 0.1

    Close = '<CLOSE>'
    High = '<HIGH>'
    Open = '<OPEN>'
    Low = '<LOW>'
    prices_name = [Close, High, Open, Low]

    @staticmethod
    def setNSteps(n_steps):
        Config.n_steps = n_steps

    @staticmethod
    def getNSteps():
        return Config.n_steps

    MAE = 'MAE'
    MSE = 'MSE'
    RMSE = 'RMSE'
    MAPE = 'MAPE'
    R2 = 'R2'
    metrics_name = [MAE, MSE, RMSE, MAPE, R2]
