import unittest
import pandas as pd
import indicators


class ModelsTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_ma(self):
        data = pd.read_excel("df_resampled.xlsx")
        df_with_ma = indicators.get_ma(data, "close", 10)
        self.assertTrue("close_SMA_10" in df_with_ma.columns)

    def test_get_rsi(self):
        data = pd.read_excel("df_resampled.xlsx")
        df_with_rsi = indicators.get_rsi(data, "close", 14)
        self.assertTrue("close_RSI_14" in df_with_rsi.columns)