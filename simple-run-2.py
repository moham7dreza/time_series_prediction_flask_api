# multivariate multi-headed 1d cnn example
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, GRU, Dropout, SimpleRNN, \
    ConvLSTM2D, concatenate
from keras.models import Model
from numpy import array
from numpy import hstack


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def cnn():
    # define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([25, 35, 45, 55, 65, 75, 85, 105, 115])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    # choose a number of time steps
    n_steps = 3
    dropout_rate = 0.2
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # separate output
    y1 = y[:, 0].reshape((y.shape[0], 1))
    y2 = y[:, 1].reshape((y.shape[0], 1))
    y3 = y[:, 2].reshape((y.shape[0], 1))
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
    # fit model
    model.fit(X, [y1, y2, y3], epochs=2000, verbose=0)
    # demonstrate prediction
    x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


def lstm():
    # define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([25, 35, 45, 55, 65, 75, 85, 105, 115])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    # choose a number of time steps
    n_steps = 3
    dropout_rate = 0.2
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # separate output
    y1 = y[:, 0].reshape((y.shape[0], 1))
    y2 = y[:, 1].reshape((y.shape[0], 1))
    y3 = y[:, 2].reshape((y.shape[0], 1))
    # define model
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
    # fit model
    model.fit(X, [y1, y2, y3], epochs=2000, verbose=0)
    # demonstrate prediction
    x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


if __name__ == '__main__':
    cnn()
    lstm()
