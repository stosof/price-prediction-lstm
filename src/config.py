import os

BASE_DIR = ".."
DATA_DIR = "data"
CURRENCY_SUBDIRS = ["eurusd", "gbpusd"]

TRAINING_START_DATE = "1/1/2017"
TRAINING_END_DATE = "1/1/2018"
RESAMPLE_TF = 15

MA_PERIODS = [5, 10, 15, 50, 100]
RSI_PERIODS = [7, 14, 21]

PIP_TARGETS = [0.001, 0.005, 0.01]
DELTA_PERIODS = [1, 3, 5, 10, 15, 25, 35, 50, 60, 90, 180, 300]

def get_currency_dir_paths():
    currency_dir_paths = []
    for subdir in CURRENCY_SUBDIRS:
        path = os.path.join(BASE_DIR, DATA_DIR, subdir)
        currency_dir_paths.append(path)
    return currency_dir_paths

