class Config:
    colab = True
    plotting = False
    date_col = "<DTYYYYMMDD>"
    start_date = '2017-06-10'
    end_date = '2022-12-03'
    epochs_for_multivariate_series = 100
    epochs_for_univariate_series = 100
    n_steps = 3
    dropout_rate = 0.2
    prediction_col = '<CLOSE>'
    local_csv_folder_path = '../../iran_stock/'
    drive_csv_folder_path = '/content/drive/My Drive/iran_stock'
    drive_model_folder_path = '/content/drive/My Drive/time_series_models'
    # 1. set csv dataset file names
    dollar_file_name = 'dollar_tjgu_from_2012.csv'
    car_file_name = 'Iran.Khodro_from_2001.csv'
    oil_file_name = 'S_Parsian.Oil&Gas_from_2012.csv'
    home_file_name = 'Maskan.Invest_from_2014.csv'
    gold_file_name = 'Lotus.Gold.Com.ETF_from_2017.csv'
    # dataset titles
    titles = ['Dollar', 'Home', 'Oil', 'Car', 'Gold',]
    Dollar = 'Dollar'
    Home = 'Home'
    Oil = 'Oil'
    Car = 'Car'
    Gold = 'Gold'
    # model names
    CNN = 'CNN'
    LSTM = 'LSTM'
    bi_LSTM = 'bi-LSTM'
    GRU = 'GRU'
    bi_GRU = 'bi-GRU'
    ANN = 'ANN'
    bi_ANN = 'bi-ANN'
    RNN = 'RNN'
    bi_RNN = 'bi-RNN'
    # series type
    univariate = 'univariate'
    multivariate = 'multivariate'

    plot_labels = ['M-CNN', 'U-CNN',
                   'M-B-LSTM', 'U-B-LSTM', 'M-LSTM', 'U-LSTM',
                   'M-GRU', 'U-GRU', 'M-B-GRU', 'U-B-GRU',
                   'M-ANN', 'U-ANN', 'M-B-ANN', 'U-N-ANN',
                   'M-RNN', 'U-RNN', 'M-B-RNN', 'U-N-RNN',
                   'REAL']
