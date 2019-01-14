import unittest
from models import LSTM_NN
import keras
import numpy as np

class ModelsTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_compiled_lstm(self):
        lstm = LSTM_NN()
        model = lstm.get_compiled_lstm()
        self.assertIsInstance(model, keras.engine.sequential.Sequential, "There was an issue creating the lstm model.")

    def test_get_training_data(self):
        lstm = LSTM_NN()
        X, Y, X_val, Y_val = lstm.get_training_data()
        self.assertIsInstance(X, np.ndarray, "There was an issue getting X training data.")
        self.assertIsInstance(Y, np.ndarray, "There was an issue getting Y training data.")
        self.assertIsInstance(X_val, np.ndarray, "There was an issue getting X training data.")
        self.assertIsInstance(Y_val, np.ndarray, "There was an issue getting Y training data.")

    def test_fit_model(self):
        lstm = LSTM_NN()
        X, Y, X_val, Y_val = lstm.get_training_data()
        history = lstm.fit_model(X=X, Y=Y, X_val=X_val, Y_val=Y_val, n_batch=1000, nb_epoch=1)
        self.assertIsInstance(history, keras.callbacks.History, "There was an issue training the model.")


