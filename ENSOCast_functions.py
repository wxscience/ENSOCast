import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import subprocess
from dateutil.relativedelta import relativedelta
import datetime as dt
import os
import yaml


def load_settings(filename):
    # This function loads settings from a config.yml file for use in ENSOCast.py.
    with open(filename, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except ImportError:
            print('The file could not be loaded.')

    return config


def get_predictors(predictors_list, start_date, end_date):
    date_list = [start_date + relativedelta(months=i) for i in range(month_diff(start_date, end_date))]
    x = np.zeros((len(date_list), len(predictors_list)))

    for predictor in predictors_list:
        if predictor == 'nino34':
            query = 'rm', 'sstoi.indices'
            subprocess.run(query)

            query = 'curl', '-O', 'http://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt'
            subprocess.run(query)
            print('SST File Downloaded')

            # nino34 starts in 1950, so we need to determine first row to keep based on that
            first_entry = month_diff(dt.date(1950, 1, 1), start_date)
            x[:, predictors_list.index(predictor)] = np.genfromtxt('detrend.nino34.ascii.txt', skip_header=1)[first_entry:, 4]

        if predictor == 'trades_wpac':
            query = 'rm', 'wpac850'
            subprocess.run(query)

            query = 'curl', '-O', 'http://www.cpc.ncep.noaa.gov/data/indices/wpac850'
            subprocess.run(query)
            print('Trade Winds File Downloaded')

            # trade winds begin in 1979
            first_entry = month_diff(dt.date(1979, 1, 1), start_date)
            wpac_data = np.genfromtxt('wpac850', skip_header=50, max_rows=41,
                                      delimiter=(4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6))[:, 1:]
            x[:, predictors_list.index(predictor)] = np.reshape(wpac_data, (wpac_data.shape[0] * 12))[first_entry:x.shape[0]]
    return x


def get_predictand(start_date, end_date):
    query = 'rm', 'sstoi.indices'
    subprocess.run(query)

    query = 'curl', '-O', 'http://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt'
    subprocess.run(query)
    print('SST File Downloaded')

    # nino34 starts in 1950, so we need to determine first row to keep based on that
    first_entry = month_diff(dt.date(1950, 1, 1), start_date)
    y_hat = np.genfromtxt('detrend.nino34.ascii.txt', skip_header=1)[first_entry:, 4]

    return y_hat


def month_diff(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month


def create_lstm(x, y, e, batch, v, s, i_nodes, h_nodes, activation_func, gpu=False):
    # Hide tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not gpu:
        # Set Keras to use CPU; CPU is sometimes faster than GPU for LSTMs
        print('Turning off GPU; Tensorflow will use CPU instead.')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print('Tensorflow will use GPU. This is not always particularly fast for LSTMs.')

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(i_nodes, activation=activation_func, return_sequences=True, input_shape=x.shape[1:]))
    model.add(Dropout(0.6))
    model.add(LSTM(h_nodes, activation=activation_func))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    # I prefer to see metrics in MAE instead of MSE so that the units are familiar.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(x, y, epochs=e, batch_size=batch, verbose=v, shuffle=s)

    return model


def make_window(data_in, window_size):
    if type(data_in) == list:  # default only 1 dim for this one
        windowed_data = np.empty((len(data_in) - window_size + 1, window_size), dtype='datetime64[s]')

        for x in range(len(data_in) - window_size + 1):
            windowed_data[x, :] = data_in[x:x + window_size]

    else:
        if data_in.ndim == 2:
            windowed_data = np.empty((len(data_in) - window_size + 1, window_size, data_in.shape[1]), dtype='float64')
        else:
            windowed_data = np.empty((len(data_in) - window_size + 1, window_size), dtype='float64')

        for x in range(len(data_in) - window_size + 1):
            if data_in.ndim == 2:
                windowed_data[x, :, :] = data_in[x:x + window_size, :]
            else:
                windowed_data[x, :] = data_in[x:x + window_size]

    return windowed_data


# You could also use sklearn's package here, but I like to practice making cool code.
class DataScaler:
    def __init__(self, the_data):
        self.min = np.min(the_data, axis=0)
        self.max = np.max(the_data, axis=0)
        self.scaled = 2 * (the_data - self.min) / (self.max - self.min) - 1

    def inverse(self, i_data):
        inverted_data = (i_data + 1) / 2 * (self.max - self.min) + self.min
        return inverted_data
