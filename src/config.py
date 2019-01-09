import os

BASE_DIR = ".."
DATA_DIR = "data"
CURRENCY_SUBDIRS = ["eurusd", "gbpusd"]

TRAINING_START_DATE = "1/1/2017"
TRAINING_END_DATE = "1/1/2018"

def get_currency_dir_paths():
    currency_dir_paths = []
    for subdir in CURRENCY_SUBDIRS:
        path = os.path.join(BASE_DIR, DATA_DIR, subdir)
        currency_dir_paths.append(path)
    return currency_dir_paths

