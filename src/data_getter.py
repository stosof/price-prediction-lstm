"""
The DataGetter class is used to read the raw input data.
We create features by adding several Moving Average and RSI indicators and calculating the change in their values
compared to several steps back in the dataset - (t-1), (t-5), (t-15) ...
We reshape the data into the 3D format required for LSTM input (samples, time_steps, features).
"""

import pandas as pd
import tf_resampler
import os
import config
import numpy as np
import indicators


class DataGetter(object):
    def get_df_base(self):
        config.DF_BASE_START_DATE = config.TRAINING_DATE_START
        config.DF_BASE_END_DATE = config.TRAINING_DATE_END
        date_range = pd.date_range(start=config.DF_BASE_START_DATE, end=config.DF_BASE_END_DATE,
                                   freq=config.DF_BASE_FREQUENCY)
        df_base = pd.DataFrame(index=date_range)
        return df_base

    def get_raw_data_from_excel_files_for_single_currency(self, currency_dir):
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

    def get_single_currency_raw_data_with_base(self, currency_dir):
        df_base = self.get_df_base()
        raw_data = self.get_raw_data_from_excel_files_for_single_currency(currency_dir)
        return df_base.join(raw_data, rsuffix=currency_dir)

    def get_tf_resampled_single_currency_raw_data_with_base(self, currency_dir):
        df_raw_base = self.get_single_currency_raw_data_with_base(currency_dir)
        df_resample = tf_resampler.resample(df_raw_base)
        return df_resample

    def get_df_with_indicators_single_currency(self, currency_dir):
        df_result = self.get_tf_resampled_single_currency_raw_data_with_base(currency_dir)
        for period in config.MA_PERIODS:
            df_result = indicators.get_ma(df_result, "close", period)
        for period in config.RSI_PERIODS:
            df_result = indicators.get_rsi(df_result, "close", period)
        return df_result

    # TODO This is currently a placeholder that returns only data from the EURUSD dir.
    def get_df_with_indicators_multicurrency(self):
        df_result = self.get_df_with_indicators_single_currency("../data/eurusd/")
        return df_result

    """"We are going to frame this as a classification problem. We will add columns that show (for each row) if the price
        first moved up or down for a given amount. We will mark these with 1 - for up and 0 - for down. We will then train
        a LSTM model to predict the class of each row in our dataset."""

    def get_targets_long_short(self):
        df_result = self.get_df_with_indicators_multicurrency()
        for target_pips in config.PIP_TARGETS:
            long_col_name = "target_long_" + str(target_pips)
            short_col_name = "target_short_" + str(target_pips)
            df_result[long_col_name] = df_result["close"] + target_pips
            df_result[short_col_name] = df_result["close"] - target_pips
        return df_result

    def get_first_reached_targets(self):
        df_result = self.get_targets_long_short()
        column_names = df_result.columns.tolist()
        long_target_col_indexes, short_target_col_indexes = self._get_indexes_of_target_columns(column_names)
        data = df_result.values
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
                    if curr_low < active_short_target:
                        list_of_checked_targets.append(0)
                        break
                    if curr_high > active_long_target:
                        list_of_checked_targets.append(1)
                        break

            checked_targets_arr = np.array(list_of_checked_targets)
            padding_amount = (df_result.count() - checked_targets_arr.size)
            checked_targets_arr = np.pad(checked_targets_arr, (0, padding_amount[0]), 'constant',
                                         constant_values=(-1, -1))
            df_result[new_col_names[list_index]] = checked_targets_arr

        return df_result

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

    def get_deltas(self):
        df_result = self.get_first_reached_targets()
        df_columns = df_result.columns
        cols_for_delta_calc = self._get_feature_columns(df_columns)

        for col in cols_for_delta_calc:
            for period in config.DELTA_PERIODS:
                new_col_name = col + "_delta_" + str(period)
                df_result[new_col_name] = df_result[col].shift(period)

        df_result = df_result.dropna()
        return df_result

    def _get_feature_columns(self, column_names):
        cols_for_delta_calc = []
        for col_name in column_names:
            if "target" in col_name or "datetime" in col_name:
                continue
            else:
                cols_for_delta_calc.append(col_name)
        return cols_for_delta_calc

    def get_standardized_and_normalized_df(self):
        df_result = self.get_deltas()
        feature_cols = self._get_feature_columns(df_result.columns)
        self._standardize_df(df_result, feature_cols)
        self._normalize_df(df_result, feature_cols)
        return df_result

    def _normalize_df(self, df, cols_to_normalize):
        for col in cols_to_normalize:
            self._normalize_col(df, col)

    def _normalize_col(self, df, col):
        col_max = df[col].max()
        col_min = df[col].min()
        max_col_name = col + "_max"
        min_col_name = col + "_min"
        df[col] = (df[col] - col_min) / (col_max - col_min)
        return max_col_name, col_max, min_col_name, col_min

    def _standardize_df(self, df, cols_to_standardize):
        for col in cols_to_standardize:
            self._standardize_col(df, col)

    def _standardize_col(self, df, col):
        col_mean = df[col].mean()
        col_std = df[col].std()
        mean_col_name = col + "_mean"
        std_col_name = col + "_std"
        df[col] = (df[col] - col_mean) / col_std
        return mean_col_name, col_mean, std_col_name, col_std

    def get_reshaped_data_for_lstm(self):
        df_result = self.get_standardized_and_normalized_df()
        df_cols = df_result.columns
        feature_cols = self._get_feature_columns(df_cols)
        data = df_result[feature_cols]
        data = data.values
        reshaped_data = []
        for row in range(len(data) - config.SEQUENCE_LENGTH):
            seq_tmp = data[row:(row + config.SEQUENCE_LENGTH)]
            reshaped_data.append(seq_tmp)
        reshaped_data = np.array(reshaped_data)
        return reshaped_data


if __name__ == "__main__":
    data_getter = DataGetter()
    data_getter.get_reshaped_data_for_lstm()
