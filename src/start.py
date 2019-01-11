import models
import config


def start_evaluation():
    model = models.LSTM_NN()
    config.TESTING_DATE_START = "1/1/2018"  # MM/DD/YYYY
    config.TESTING_DATE_END = "31/3/2018"
    config.TESTING_MODEL_ARCHITECTURE = "model_architecture.json"
    config.TESTING_MODEL_WEIGHTS = "weights-improvement-99-0.80.hdf5"
    model.start_model_evaluation()


def start_training():
    model = models.LSTM_NN()
    config.TRAINING_DATE_START = "1/1/2017"
    config.TRAINING_DATE_END = "1/1/2018"
    model.start_model_training()


if __name__ == "__main__":
    # start_training()
    start_evaluation()
