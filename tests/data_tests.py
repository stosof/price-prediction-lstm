import unittest
import config
import os

class DataExistsInInputDirsTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_subdirs_exist_in_data_dir(self):
        currency_dir_paths = config.get_currency_dir_paths()
        for dir in currency_dir_paths:
            self.assertTrue(os.path.isdir(dir), "The input data directory - {} - defined in config.py does not exist.".format(dir))

    def _check_contains_xlsx(self, iterable):
        for element in iterable:
            if ".xlsx" in element:
                return True
        return False

    def _check_contains_other(self, iterable):
        for element in iterable:
            if ".xlsx" not in element:
                return True
        return False

    def test_data_subdirs_contain_xlsx_files(self):
        currency_dir_paths = config.get_currency_dir_paths()
        for dir in currency_dir_paths:
            files_in_dir = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            self.assertTrue(self._check_contains_xlsx(files_in_dir), "The directory - {} - does not contain any .xlsx files".format(dir))
            self.assertFalse(self._check_contains_other(files_in_dir), "The directory - {} - contains files that are not .xlsx files".format(dir))

if __name__ == '__main__':
    unittest.main()
