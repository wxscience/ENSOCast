import time as t
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import ENSOCast_functions as ef
from dateutil.relativedelta import relativedelta

config = ef.load_settings('config.yml')

# Get the arguments for the file paths/names
image_path = config['image_path']
image_name = config['image_name']
predictors = config['predictors']

# quickly assign variables to the dictionary that represents model_config. Is there a more clever way to do this?
for key, value in config['model_config'].items():
    exec(key + '=value')

# convert dates
start_date, end_date = dt.datetime.strptime(start_date, '%Y-%m-%d'), dt.datetime.strptime(end_date, '%Y-%m-%d')

if real_time_indicator:
    end_date = dt.datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

verbosity = 0  # verbosity level (1 for status bar, 2 for more info, 0 for none)
shuffle = False  # shuffle?

# Load in the specified predictors. Right now there's only 2 possible, but there will be more in future editions.
# Create list of dates so we know how long to make each column.
x = ef.get_predictors(predictors, start_date, end_date)
y = ef.get_predictand(start_date, end_date)  # y_hat is really just sst34 which can also double as a predictor

# Everything below here is not to be configured. #
# Create date "list" for *DATE VALID* (not the same as date initialized)
dates = [start_date + relativedelta(months=i) for i in range(ef.month_diff(start_date, end_date))]

# SSTs are updated on the 7th of each month so safest to wait until the 8th
if (dates[-1].month == dt.datetime.now().month-1) and (dt.datetime.now().day < 8):
    del dates[-1]

# Normalize data to be between -1 and 1. Hold on to data_obj so that we can transform back afterwards.
x_obj = ef.DataScaler(x)
x_scaled = x_obj.scaled

y_obj = ef.DataScaler(y)
y_scaled = y_obj.scaled

# We train on smoothed data and test on unsmoothed data
x_train, x_test = x_scaled[0:train_len], x_scaled[train_len:]
y_train, y_test = y_scaled[0:train_len], y_scaled[train_len:]

dates_test = dates[train_len:]  # we have no use for dates_train

# Realtime data, dates_realtime is 1 year longer b/c it includes forecast valid dates
x_realtime = x_scaled[-num_back_plot*2:]
y_realtime = y_scaled[-num_back_plot*2:]

dates_realtime = np.asarray([start_date + relativedelta(months=i)
                             for i in range(y.shape[0]+13)])[-num_back_plot-12:]

# Split into x and y, also Keras requires dimensions of (samples x time_step x features=window_length)
x_train_windowed = ef.make_window(x_train, window_length)[:-fcst_size, :]
y_train_windowed = ef.make_window(y_train[window_length:], fcst_size)

x_test_windowed = ef.make_window(x_test, window_length)[:-fcst_size, :]
y_test_windowed = ef.make_window(y_test[window_length:], fcst_size)

x_realtime_windowed = ef.make_window(x_realtime, window_length)

# this will be a list of dates that corresponds exactly with y_test
dates_y_test = ef.make_window(dates_test[window_length:], fcst_size)

# Pre-define y_hat:
y_hat = np.zeros(y_test_windowed.shape)
y_hat_realtime = np.zeros((len(x_realtime_windowed), fcst_size))  # always forecast 12 months in future

# Keeping time is interesting, especially when testing CPU vs. GPU (CPU seems to win with small LSTMs; GPU with larger)
t1 = t.time()

for n_fcst in range(fcst_size):
    # Train the Neural Network
    enso_model = ef.create_lstm(x_train_windowed, y_train_windowed[:, n_fcst], epochs, batch_size, verbosity, shuffle,
                                initial_nodes, hidden_nodes, activation_function, gpu)
    y_hat[:, n_fcst] = np.squeeze(enso_model.predict(x_test_windowed))
    y_hat_realtime[:, n_fcst] = np.squeeze(enso_model.predict(x_realtime_windowed))


time_elapsed = t.time() - t1
print(str(time_elapsed) + ' seconds')

# Untransform so that the values are correct.
y_test_windowed = y_obj.inverse(y_test_windowed)
y_hat = y_obj.inverse(y_hat)
y_hat_realtime = y_obj.inverse(y_hat_realtime)

# We don't need to loop through the dates and y_test. They are *valid* dates, so they remain static.
dates_plot = dates_y_test[fcst_size-1:, 0]
y_test_plot = y_test_windowed[fcst_size-1:, 0]

# Get the red and blue shaded areas
y_test_pos = np.copy(y_test_plot)  # y_test at 0 is the 0 month forecast = analysis
y_test_pos[y_test_pos < 0] = np.nan

y_test_neg = np.copy(y_test_plot)
y_test_neg[y_test_neg > 0] = np.nan

# Plotting fcst_valid dates, not fcst_init dates.
for n in [0, int(fcst_size/2)]:
    plt.figure(figsize=(3, 4), dpi=300)
    for i in range(n, n+int(fcst_size/2)):
        y_hat_plot = y_hat[(fcst_size-1-i):-i or None, i]

        r = np.corrcoef(y_hat_plot, y_test_plot)[0, 1]
        MSE = np.sqrt((y_hat_plot - y_test_plot) ** 2).mean()

        ax = plt.subplot(fcst_size/2, 1, i+1-n)
        plt.axis([dates_plot[0], dates_plot[-1], -3, 3])
        plt.ylim((-3, 3))
        ax.set_yticks((-3, -2, -1, 0, 1, 2, 3))

        ax.fill_between(dates_plot, 0, y_test_pos, facecolor='red', interpolate=False, alpha=0.5)
        ax.fill_between(dates_plot, 0, y_test_neg, facecolor='blue', interpolate=False, alpha=0.5)

        # plt.plot(date_list, y_test, color='black', linewidth=1)
        plt.plot(dates_plot, np.zeros(len(dates_plot)), color='black', linewidth=1)
        plt.plot(dates_plot, y_hat_plot, color='#000000', linewidth=1)

        plt.xticks(fontsize=3, weight='bold')
        plt.yticks(fontsize=3, weight='bold')
        plt.text(dates_plot[1], 2, ' r = ' + str(np.around(r, decimals=2)) + ' | RMSE = '
                 + str(np.around(MSE, decimals=2)), size=3)
        plt.ylabel('SST 3.4', size=4, weight='bold')

        if i == 5:
            plt.xlabel('Verification from  www.KyleMacRitchie.com/ENSO', size=3, color='red', weight='bold')

        plt.title(str(i + 1) + ' Month Forecast', size=4, weight='bold')

    plt.subplots_adjust(hspace=1)
    plt.suptitle('ENSOCast: SST 3.4 Monthly Forecast Verification', size=5, weight='bold')
    plt.savefig(image_path + image_name + '_verif_'+str(n)+'.png')
    plt.close()

# Make the forecast plots.
plt.figure(figsize=(3, 4.5), dpi=300)
for i in range(6):
    y_plot = np.zeros(24)
    y_plot[:] = np.nan
    ax = plt.subplot(6, 1, i + 1)
    plt.axis([dates_realtime[0], dates_realtime[-1], -2, 2])
    plt.ylim((-2, 2))
    ax.set_yticks((-2, -1, 0, 1, 2))

    t1 = y_realtime[-num_back_plot:len(y_realtime)-i]
    t2 = y_hat_realtime[-1-i, :]  # plot from end to beginning since most recent plot is first

    dates_t1 = dates_realtime[0:len(y_realtime)-num_back_plot-i]
    dates_t2 = dates_realtime[len(y_realtime)-num_back_plot-i:]

    y_plot[0:24-i] = np.concatenate((t1, t2))

    # Plot the + and - 0.5 lines
    plt.plot(dates_realtime, 0.5 + np.zeros(24), linewidth=0.5, color='red', alpha=0.5)
    plt.plot(dates_realtime, np.zeros(24) - 0.5, linewidth=0.5, color='blue', alpha=0.5)

    plt.plot(dates_t1, t1, color='#000000', marker='o', markersize=1.5, linewidth=0)
    plt.plot(dates_t2, y_plot[len(t1):], marker='o', markersize=1.5, color='red', linewidth=0)

    plt.xticks(fontsize=3, weight='bold')
    plt.yticks(fontsize=3, weight='bold')
    plt.ylabel('SST 3.4', size=4, weight='bold')

    if i == 5:
        plt.xlabel('Created: ' + dt.datetime.now().strftime('%B %d %Y') + ' | www.KyleMacRitchie.com/ENSO',
                   size=3, color='red', weight='bold')

    plt.title('Initialized: ' + dates_realtime[11 - i].strftime('%B %Y'), size=4, weight='bold')
    plt.grid(False)

plt.subplots_adjust(hspace=.8)
plt.suptitle('ENSOCast: SST 3.4 Anom (' + chr(176) + 'C) 6-Month Forecasts', size=5, weight='bold')
plt.savefig(image_path + image_name + 'ENSOCast.png', orientation='portrait', bbox_inches='tight')
plt.close()
