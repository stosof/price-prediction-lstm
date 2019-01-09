"""
Read the training dataset. We do not have any volume data so we drop the volume column.
We set the datetime as the index of the DF to allow for easier timeseries operations.
"""

import pandas as pd
import tf_resampler
import talib
import os
import config
import numpy as np


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

    def get_tf_resampled_single_currency_raw_data_with_base(self, start_date, end_date, currency_dir):
        df_raw_base = self.get_single_currency_raw_data_with_base(start_date, end_date, currency_dir)
        df_resample = tf_resampler.resample(df_raw_base)
        return df_resample

    @staticmethod
    def get_ma(df, col_name, period):
        new_col_name = col_name + "_SMA_" + str(period)
        df[new_col_name] = talib.SMA(df[col_name], period)
        df = df.dropna()
        return df

    @staticmethod
    def get_rsi(df, col_name, period):
        new_col_name = col_name + "_RSI_" + str(period)
        df[new_col_name] = talib.RSI(df[col_name], period)
        df = df.dropna()
        return df

    def get_df_with_indicators_single_currency(self, currency_dir):
        df_result = self.get_tf_resampled_single_currency_raw_data_with_base(config.TRAINING_START_DATE,
                                                                             config.TRAINING_END_DATE, currency_dir)
        for period in config.MA_PERIODS:
            df_result = self.get_ma(df_result, "close", period)
        for period in config.RSI_PERIODS:
            df_result = self.get_rsi(df_result, "close", period)
        return df_result

    def get_df_with_indicators_multicurrency(self):
        pass

    """"We are going to frame this as a classification problem. We will add columns that show (for each row) if the price
        first moved up or down for a given amount. We will mark these with 1 - for up and 0 - for down. We will then train
        a LSTM model to predict the class of each row in our dataset."""

    def get_targets_long_short(self, df, target_pips):
        long_col_name = "target_long_" + str(target_pips)
        short_col_name = "target_short_" + str(target_pips)
        df[long_col_name] = df["close"] + target_pips
        df[short_col_name] = df["close"] - target_pips
        return df

    def _get_indexes_of_target_columns(self, col_names):
        target_cols_long = []
        for col_name in col_names:
            if "target_long" in col_name:
                target_cols_long.append(col_name)

        target_cols_short = []
        for col_name in col_names:
            if "target_short" in col_name:
                target_cols_short.append(col_name)

        long_target_positions = []
        for target_col in target_cols_long:
            long_target_positions.append(col_names.index(target_col))

        short_target_positions = []
        for target_col in target_cols_short:
            short_target_positions.append(col_names.index(target_col))

        return long_target_positions, short_target_positions


    def get_first_reached_targets(self, df):
        column_names = df.columns.tolist()
        long_target_col_indexes, short_target_col_indexes = self._get_indexes_of_target_columns(column_names)
        data = df.values

        df_result = df
        checked_target_columns_ls = []
        new_col_names = []
        for counter, target in enumerate(long_target_col_indexes):
            checked_target_columns_ls.append([])
            target_check_col_name = "target_" + str(counter)
            new_col_names.append(target_check_col_name)

        for list_index, list_of_checked_targets in enumerate(checked_target_columns_ls):
            for row_to_check_reached_target in range(len(data)):
                active_long_target = data[row_to_check_reached_target][long_target_col_indexes[list_index]]
                active_short_target = data[row_to_check_reached_target][short_target_col_indexes[list_index]]
                for row_to_compare_with in range(len(data) - row_to_check_reached_target):
                    curr_high = data[row_to_check_reached_target + row_to_compare_with][2]
                    curr_low = data[row_to_check_reached_target + row_to_compare_with][3]
                    if (curr_low < active_short_target):
                        list_of_checked_targets.append(0)
                        break
                    if (curr_high > active_long_target):
                        list_of_checked_targets.append(1)
                        break

            checked_targets_arr = np.array(list_of_checked_targets)
            padding_amount =  (df_result.count() - checked_targets_arr.size)
            checked_targets_arr = np.pad(checked_targets_arr, (0,padding_amount[0]), 'constant', constant_values=(np.NaN, np.NaN))
            df_result[new_col_names[list_index]] = checked_targets_arr

        return df_result

    def _get_columns_for_delta_calculation(self, column_names):
        cols_for_delta_calc = []
        for col_name in column_names:
            if "target" in col_name or "datetime" in col_name:
                continue
            else:
                cols_for_delta_calc.append(col_name)
        return cols_for_delta_calc

    def get_deltas(self, df):
        df_columns = df.columns
        cols_for_delta_calc = self._get_columns_for_delta_calculation(df_columns)

        for col in cols_for_delta_calc:
            for period in config.DELTA_PERIODS:
                new_col_name = col + "_delta_" + str(period)
                df[new_col_name] = df[col].shift(period)

        df = df.dropna()
        return df

    def standardize_and_normalize_df(self):
        pass


if __name__ == "__main__":
    data_getter = DataGetter()
    data_getter.get_single_currency_raw_data_from_all_excel_files("../data/eurusd")
