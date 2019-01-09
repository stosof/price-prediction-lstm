"""
Read the training dataset. We do not have any volume data so we drop the volume column.
We set the datetime as the index of the DF to allow for easier timeseries operations.
"""

import pandas as pd
import tf_resampler
import talib

class DataGetter(object):

    def get_df_base(self, start_date, end_date, frequency):
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        df_base = pd.DataFrame(index=date_range)
        return df_base

    def get_single_currency_raw_data_from_excel(self, excel_file_path):
        training_data = pd.read_excel(excel_file_path, names=["datetime", "open", "high", "low", "close", "volume"])
        training_data.index = pd.to_datetime(training_data.datetime, unit='s')
        training_data = training_data[["open", "high", "low", "close"]]
        return training_data

    def get_single_currency_raw_data_with_base(self, start_date, end_date, interval, excel_file_path, suffix):
        df_base = self.get_df_base(start_date, end_date, interval)
        raw_data = self.get_single_currency_raw_data_from_excel(excel_file_path)
        return df_base.join(raw_data, rsuffix=suffix) # Join on index of DF

    def get_tf_resampled_single_currency_raw_data_with_base(self, start_date, end_date, interval, excel_file_path,
                                                            suffix, tf):
        df_raw_base = self.get_single_currency_raw_data_with_base(start_date, end_date, interval, excel_file_path, suffix)
        df_resample = tf_resampler.resample(df_raw_base, tf)
        return df_resample

    def get_ma(self, df, col_name, period):
        new_col_name = col_name + "_SMA_" + str(period)
        df[new_col_name] = talib.SMA(df[col_name], period)
        df = df.dropna()
        return df
