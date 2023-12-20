from numpy import array

from pandora.Config.Config import Config


class DataSampler:
    # split a multivariate sequence into samples
    @staticmethod
    def split_sequences(series_type, sequences, n_steps=Config.n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences) - 1:
                break
            # gather input and output parts of the pattern
            if series_type == 'multivariate':
                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
            elif series_type == 'univariate':
                seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
            else:
                raise Exception('Unknown series type')
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
