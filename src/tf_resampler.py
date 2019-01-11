import pandas as pd
import config


# This mapping needs to be updated for the different TFs we want to support.
def get_datetime_delta(new_datetime, new_tf, m1_datetimes):
    if new_datetime == 0:
        return m1_datetimes[0]
    # Mapped TFs
    if new_tf == 5:
        return new_datetime + pd.Timedelta("5min")
    if new_tf == 15:
        return new_datetime + pd.Timedelta("15min")


def resample(m1_data):
    m1_arr = m1_data.values
    m1_datetimes = m1_data.index
    new_open = 0
    new_high = 0
    new_low = 1000000
    new_close = 0
    new_datetime = 0
    counter = 1
    resampled_data = []
    for pos, row in enumerate(m1_arr):
        if counter == config.RESAMPLE_TF:
            new_close = row[3]
        if counter > config.RESAMPLE_TF:
            counter = 1
            resampled_data.append([new_datetime, new_open, new_high, new_low, new_close])
            new_high = 0
            new_low = 1000000
        if counter == 1:
            new_datetime = get_datetime_delta(new_datetime, config.RESAMPLE_TF, m1_datetimes)
            new_open = row[0]
        if row[1] > new_high:
            new_high = row[1]
        if row[2] < new_low:
            new_low = row[2]
        counter += 1

    resampled_data = pd.DataFrame(resampled_data, columns=["datetime", "open", "high", "low", "close"])
    resampled_data = resampled_data.dropna()
    return resampled_data
