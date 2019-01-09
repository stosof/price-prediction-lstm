import unittest
import config
from data_getter import DataGetter
import pandas as pd


class DataGetterTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_df_base(self):
        data_getter = DataGetter()
        df_base = data_getter.get_df_base("1/1/2017", "1/1/2018")
        self.assertIsInstance(df_base, pd.DataFrame, "There was an issue creating df_base.")

    def test_get_single_currency_raw_data_from_excel(self):
        data_getter = DataGetter()
        for currency_dir in config.get_currency_dir_paths():
            df_raw = data_getter.get_single_currency_raw_data_from_all_excel_files(currency_dir)
            self.assertIsInstance(df_raw, pd.DataFrame,
                                  "There was an issue creating a dataframe from a file in - {}.".format(currency_dir))

    def test_get_single_currency_raw_data_with_base(self):
        data_getter = DataGetter()
        for currency_dir in config.get_currency_dir_paths():
            df_base_raw = data_getter.get_single_currency_raw_data_with_base(config.TRAINING_START_DATE,
                                                                             config.TRAINING_END_DATE, currency_dir)
            self.assertIsInstance(df_base_raw, pd.DataFrame,
                                  "There was an issue creating df_base_raw from the files in dir - {dir}.".format(
                                      dir=dir))

    def test_tf_resampled_single_currency_raw_data_with_base(self):
        data_getter = DataGetter()
        for currency_dir in config.get_currency_dir_paths():
            df_resampled = data_getter.get_tf_resampled_single_currency_raw_data_with_base(config.TRAINING_START_DATE,
                                                                                           config.TRAINING_END_DATE,
                                                                                           currency_dir, 15)
            self.assertIsInstance(df_resampled, pd.DataFrame,
                                  "There was an issue creating df_resampled from dir - {dir}.".format(dir=currency_dir))

    def test_get_ma(self):
        data = pd.read_excel("df_resampled.xlsx")
        data_getter = DataGetter()
        df_with_ma = data_getter.get_ma(data, "close", 10)
        self.assertTrue("close_SMA_10" in df_with_ma.columns)

    def test_get_rsi(self):
        data = pd.read_excel("df_resampled.xlsx")
        data_getter = DataGetter()
        df_with_rsi = data_getter.get_rsi(data, "close", 14)
        self.assertTrue("close_RSI_14" in df_with_rsi.columns)

    def test_get_df_with_indicators_single_currency(self):
        data_getter = DataGetter()
        for currency_dir in config.get_currency_dir_paths():
            df_result = data_getter.get_df_with_indicators_single_currency(currency_dir)
            self.assertIsInstance(df_result, pd.DataFrame,
                                  "There was an issue creating the df_result with the data in this dir - {dir}".format(
                                      dir=currency_dir))


if __name__ == '__main__':
    unittest.main()
