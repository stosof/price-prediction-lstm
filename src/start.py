""" The start module is used as the entry point for inference and training jobs.

Both inference and training jobs are configured using the config module. The functions in the start module set the
necessary parameters in the config module and call their respective processes.

Data for both inference and training is stored in the data folder. Data is organized by financial instrument (currency
for this example). The format is expected to be Excel files containing OHLC data.

This example is configured to train and evaluate the accuracy of a LSTM Neural Network.
We represent the price prediction problem as a classification task.
For each row in out data we calculate two targets => [close_price + target_pips] and [close_price - target_pips].
Target pips are defined in config.PIP_TARGETS.
We then check which target the price will move to first and add a binary classification column:
    - 1 = we reached the long target first
    - 0 = we reached the short target first.

start_evaluation() - Starts the inference process.
Generates predictions for the period between TESTING_DATE_START and TESTING_DATE_END.
Loads the saved model architecture with the name of TESTING_MODEL_ARCHITECTURE from the models dir.
Loads the weights to be used with the specified model architecture from TESTING_MODEL_WEIGHTS.
    Output:
        - Classification report printed to stdout and logged in /output/price-prediction-lstm.log
        - Transformed input data with added predictions in /output/df_predictions.xlsx

start_training() - Starts the training process.
Trains on data from TRAINING_DATE_START to TRAINING_DATE_END.
Saves the trained model's architecture and weights to /output/ dir.

"""

import models
import config


def start_evaluation():
    config.MODE = "test"
    model = models.LSTM_NN()
    config.TESTING_DATE_START = "2/1/2018"  # MM/DD/YYYY
    config.TESTING_DATE_END = "3/1/2018"
    config.TESTING_MODEL_ARCHITECTURE = "model_architecture.json"
    config.TESTING_MODEL_WEIGHTS = "weights-improvement-109-0.79.hdf5"
    model.start_model_evaluation()


def start_training():
    config.MODE = "train"
    model = models.LSTM_NN()
    config.TRAINING_DATE_START = "1/1/2017"
    config.TRAINING_DATE_END = "1/1/2018"
    config.VALIDATION_DATE_START =  "1/1/2018"
    config.VALIDATION_DATE_END = "2/1/2018"
    config.TRAINING_EPOCHS = 250
    model.start_model_training()


if __name__ == "__main__":
    # start_training()
    start_evaluation()
