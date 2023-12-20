from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, GRU, Dropout, SimpleRNN
from keras.models import Model

from pandora.Config.Config import Config


class ModelBuilder:
    @staticmethod
    def get_multi_output_stacked_LSTM_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        lstm = LSTM(100, activation='relu', return_sequences=True)(visible)
        lstm = Dropout(dropout_rate)(lstm)
        ##
        lstm = LSTM(100, activation='relu')(lstm)
        lstm = Dropout(dropout_rate)(lstm)
        lstm = Flatten()(lstm)
        lstm = Dense(50, activation='relu')(lstm)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(lstm) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_LSTM_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        bi_lstm = Bidirectional(LSTM(100, activation='relu', return_sequences=True))(visible)
        bi_lstm = Dropout(dropout_rate)(bi_lstm)
        ##
        bi_lstm = Bidirectional(LSTM(100, activation='relu'))(bi_lstm)
        bi_lstm = Dropout(dropout_rate)(bi_lstm)
        bi_lstm = Flatten()(bi_lstm)
        bi_lstm = Dense(50, activation='relu')(bi_lstm)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(bi_lstm) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_GRU_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        gru = GRU(100, activation='relu', return_sequences=True)(visible)
        gru = Dropout(dropout_rate)(gru)
        ##
        gru = GRU(100, activation='relu')(gru)
        gru = Dropout(dropout_rate)(gru)
        gru = Flatten()(gru)
        gru = Dense(50, activation='relu')(gru)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(gru) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_GRU_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        bi_gru = Bidirectional(GRU(100, activation='relu', return_sequences=True))(visible)
        bi_gru = Dropout(dropout_rate)(bi_gru)
        ##
        bi_gru = Bidirectional(GRU(100, activation='relu'))(bi_gru)
        bi_gru = Dropout(dropout_rate)(bi_gru)
        bi_gru = Flatten()(bi_gru)
        bi_gru = Dense(50, activation='relu')(bi_gru)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(bi_gru) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_RNN_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        rnn = SimpleRNN(100, activation='relu', return_sequences=True)(visible)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = SimpleRNN(100, activation='relu')(rnn)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = Flatten()(rnn)
        rnn = Dense(50, activation='relu')(rnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(rnn) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_bi_RNN_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        rnn = Bidirectional(SimpleRNN(100, activation='relu', return_sequences=True))(visible)
        rnn = Dropout(dropout_rate)(rnn)
        ##
        rnn = Bidirectional(SimpleRNN(100, activation='relu'))(rnn)
        rnn = Dropout(dropout_rate)(rnn)
        rnn = Flatten()(rnn)
        rnn = Dense(50, activation='relu')(rnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(rnn) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_ANN_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        visible = Input(shape=(n_steps, n_features))
        ann = Dense(100, activation='relu')(visible)
        ann = Dropout(dropout_rate)(ann)
        ##
        ann = Dense(100, activation='relu')(ann)
        ann = Dropout(dropout_rate)(ann)
        ann = Flatten()(ann)
        ann = Dense(50, activation='relu')(ann)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(ann) for _ in range(n_features)]
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_CNN_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        # define model
        visible = Input(shape=(n_steps, n_features))
        cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)
        cnn = Dropout(dropout_rate)(cnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(cnn) for _ in range(n_features)]
        # tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    # In Keras, the Bidirectional layer is commonly used with recurrent layers like LSTM, GRU, or SimpleRNN. However,
    # it is not directly compatible with dense (fully connected) layers. If you want to create a Bidirectional
    # architecture for an ANN, you can consider using Bidirectional with a recurrent layer followed by dense layers.

    @staticmethod
    def get_multi_output_bi_ANN_model(n_features, n_steps=Config.n_steps, dropout_rate=Config.dropout_rate):
        # Define model
        visible = Input(shape=(n_steps, n_features))

        # Bidirectional RNN layer with dropout
        rnn = Bidirectional(SimpleRNN(50, activation='relu', return_sequences=True))(visible)
        rnn = Dropout(dropout_rate)(rnn)

        # Flatten layer to connect with dense layers
        flattened_rnn = Flatten()(rnn)

        # Dense layers with dropout
        dense1 = Dense(64, activation='relu')(flattened_rnn)
        dense1 = Dropout(dropout_rate)(dense1)

        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(dropout_rate)(dense2)

        # Define outputs dynamically based on n_features
        outputs = [Dense(1)(dense2) for _ in range(n_features)]

        # Tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        return model

    @staticmethod
    def getModel(model_name, n_features):
        if model_name == "CNN":
            model = ModelBuilder.get_multi_output_CNN_model(n_features)
        elif model_name == "LSTM":
            model = ModelBuilder.get_multi_output_stacked_LSTM_model(n_features)
        elif model_name == "bi-LSTM":
            model = ModelBuilder.get_multi_output_bi_LSTM_model(n_features)
        elif model_name == "RNN":
            model = ModelBuilder.get_multi_output_RNN_model(n_features)
        elif model_name == "bi-RNN":
            model = ModelBuilder.get_multi_output_bi_RNN_model(n_features)
        elif model_name == "GRU":
            model = ModelBuilder.get_multi_output_GRU_model(n_features)
        elif model_name == "bi-GRU":
            model = ModelBuilder.get_multi_output_bi_GRU_model(n_features)
        elif model_name == "ANN":
            model = ModelBuilder.get_multi_output_ANN_model(n_features)
        elif model_name == "bi-ANN":
            model = ModelBuilder.get_multi_output_bi_ANN_model(n_features)
        else:
            raise Exception("model name not recognized")
        return model
