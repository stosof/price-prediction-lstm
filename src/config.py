import os

BASE_DIR = ".."
DATA_DIR = "data"
CURRENCY_SUBDIRS = ["eurusd", "gbpusd"]
MODELS_DIR = "models"

TRAINING_DATE_START = "1/1/2017"
TRAINING_DATE_END = "1/1/2018"
RESAMPLE_TF = 15

MA_PERIODS = [5, 10, 15, 50, 100]
RSI_PERIODS = [7, 14, 21]

PIP_TARGETS = [0.001, 0.005, 0.01]
DELTA_PERIODS = [1, 3, 5, 10, 15, 25, 35, 50, 60, 90, 180, 300]
SEQUENCE_LENGTH = 10

TRAINING_DATA_TARGET = "target_0"
TRAINING_BATCH_SIZE = 1000
TRAINING_EPOCHS = 250

TESTING_DATE_START = "1/1/2018"
TESTING_DATE_END = "3/31/2018"

DF_BASE_START_DATE = None
DF_BASE_END_DATE = None
DF_BASE_FREQUENCY = "min"

def get_currency_dir_paths():
    currency_dir_paths = []
    for subdir in CURRENCY_SUBDIRS:
        path = os.path.join(BASE_DIR, DATA_DIR, subdir)
        currency_dir_paths.append(path)
    return currency_dir_paths

def get_models_dir():
    path = os.path.join(BASE_DIR, MODELS_DIR)
    return path

