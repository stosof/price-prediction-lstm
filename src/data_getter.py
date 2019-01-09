"""
Read the training dataset. We do not have any volume data so we drop the volume column.
We set the datetime as the index of the DF to allow for easier timeseries operations.
"""

import pandas as pd
import tf_resampler
import talib
import os
import config


class DataGetter(object):
    def get_df_base(self, start_date, end_date, frequency="min"):
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        df_base = pd.DataFrame(index=date_range)
        return df_base

    def get_single_currency_raw_data_from_all_excel_files(self, currency_dir):
        files_in_dir = [f for f in os.listdir(currency_dir) if os.path.isfile(os.path.join(currency_dir, f))]

        df_ls = []
        for file in files_in_dir:
            filepath = os.path.join(currency_dir, file)
            df_tmp = pd.read_excel(filepath, names=["datetime", "open", "high", "low", "close", "volume"])
            df_ls.append(df_tmp)

        df_result = pd.concat(df_ls)
        df_result.index = pd.to_datetime(df_result.datetime, unit='s')
        df_result = df_result[["open", "high", "low", "close"]]
        return df_result

    def get_single_currency_raw_data_with_base(self, start_date, end_date, currency_dir):
        df_base = self.get_df_base(start_date, end_date)
        raw_data = self.get_single_currency_raw_data_from_all_excel_files(currency_dir)
        return df_base.join(raw_data, rsuffix=currency_dir)  # Join on index of DF

    def get_tf_resampled_single_currency_raw_data_with_base(self, start_date, end_date, currency_dir, tf):
        df_raw_base = self.get_single_currency_raw_data_with_base(start_date, end_date, currency_dir)
        df_resample = tf_resampler.resample(df_raw_base, tf)
        return df_resample

    def get_ma(self, df, col_name, period):
        new_col_name = col_name + "_SMA_" + str(period)
        df[new_col_name] = talib.SMA(df[col_name], period)
        df = df.dropna()
        return df

    def get_rsi(self, df, col_name, period):
        new_col_name = col_name + "_RSI_" + str(period)
        df[new_col_name] = talib.RSI(df[col_name], period)
        df = df.dropna()
        return df

    def get_df_with_indicators_single_currency(self, currency_dir):
        df_result = self.get_tf_resampled_single_currency_raw_data_with_base(config.TRAINING_START_DATE,
                                                                             config.TRAINING_END_DATE, currency_dir, 15)
        for period in config.MA_PERIODS:
            df_result = self.get_ma(df_result, "close", period)
        for period in config.RSI_PERIODS:
            df_result = self.get_rsi(df_result, "close", period)
        return df_result


if __name__ == "__main__":
    data_getter = DataGetter()
    data_getter.get_single_currency_raw_data_from_all_excel_files("../data/eurusd")
