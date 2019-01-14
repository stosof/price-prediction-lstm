import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from data_getter import DataGetter
import config
import os
from keras.models import model_from_json
from sklearn.metrics import classification_report
import logger


class LSTM_NN(object):
    def __init__(self):
        self.X = None
        self.X_val = None

    def get_compiled_lstm(self):
        model = Sequential()
        model.add(LSTM(1024, input_shape=(config.SEQUENCE_LENGTH, 156)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_training_data(self):
        X, X_val = self._get_training_data_x()
        Y, Y_val = self._get_training_data_y()
        return X, Y, X_val, Y_val

    def get_testing_data(self):
        X = self._get_testing_data_x()
        Y = self._get_testing_data_y()
        return X, Y

    def _get_training_data_x(self):
        data_getter = DataGetter()

        config.DF_BASE_START_DATE = config.TRAINING_DATE_START
        config.DF_BASE_END_DATE = config.TRAINING_DATE_END
        X_train = data_getter.get_reshaped_data_for_lstm()
        self.X = X_train

        config.DF_BASE_START_DATE = config.VALIDATION_DATE_START
        config.DF_BASE_END_DATE = config.VALIDATION_DATE_END
        X_val = data_getter.get_reshaped_data_for_lstm()
        self.X_val = X_val

        return X_train, X_val

    def _get_testing_data_x(self):
        data_getter = DataGetter()
        config.DF_BASE_START_DATE = config.TESTING_DATE_START
        config.DF_BASE_END_DATE = config.TESTING_DATE_END
        reshaped_data_lstm = data_getter.get_reshaped_data_for_lstm()
        self.X = reshaped_data_lstm
        return reshaped_data_lstm

    def _get_training_data_y(self):
        data_getter = DataGetter()

        config.DF_BASE_START_DATE = config.TRAINING_DATE_START
        config.DF_BASE_END_DATE = config.TRAINING_DATE_END
        df_result = data_getter.get_deltas()
        df_result = df_result[config.TRAINING_DATA_TARGET]
        y = df_result.values
        if self.X is None:
            raise Exception('X needs to be defined before defining Y. Run _get_training_data_x before this method.')
        y = y[0:self.X.shape[0]]
        y = self._one_hot_encode(y, 2)

        config.DF_BASE_START_DATE = config.VALIDATION_DATE_START
        config.DF_BASE_END_DATE = config.VALIDATION_DATE_END
        df_result = data_getter.get_deltas()
        df_result = df_result[config.TRAINING_DATA_TARGET]
        y_val = df_result.values
        if self.X_val is None:
            raise Exception('X needs to be defined before defining Y. Run _get_training_data_x before this method.')
        y_val = y_val[0:self.X_val.shape[0]]
        y_val = self._one_hot_encode(y_val, 2)

        return y, y_val

    def _get_testing_data_y(self):
        data_getter = DataGetter()
        df_result = data_getter.get_deltas()
        df_result = df_result[config.TRAINING_DATA_TARGET]
        y = df_result.values
        if self.X is None:
            raise Exception('X needs to be defined before defining Y. Run _get_testing_data_x before this method.')
        y = y[0:self.X.shape[0]]
        y = self._one_hot_encode(y, 2)
        return y

    def _one_hot_encode(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def fit_model(self, X, Y, X_val, Y_val, n_batch, nb_epoch):
        model = self.get_compiled_lstm()
        callbacks = self._get_callbacks()
        history = model.fit(X, Y, epochs=nb_epoch, batch_size=n_batch, verbose=1, shuffle=False, callbacks=callbacks,
                            validation_data=(X_val, Y_val))
        self.persist_model_json(model)
        return history

    def persist_model_json(self, model):
        model_json = model.to_json()
        models_dir = config.get_models_dir()
        models_dir = os.path.join(models_dir, "model_architecture.json")
        with open(models_dir, "w") as json_file:
            json_file.write(model_json)

    def _get_callbacks(self):
        models_dir = config.get_models_dir()
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        models_dir = os.path.join(models_dir, filepath)
        checkpoint = ModelCheckpoint(models_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list

    def start_model_training(self):
        X, Y, X_val, Y_val = self.get_training_data()
        self.fit_model(X=X, Y=Y, X_val=X_val, Y_val=Y_val, n_batch=config.TRAINING_BATCH_SIZE,
                       nb_epoch=config.TRAINING_EPOCHS)

    def _evaluate_model(self, model_json, model_h5, X, Y):
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_h5)
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(X, Y, verbose=1)
        logger.log_info("Loaded model accuracy - %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        Y_pred = loaded_model.predict_classes(X)
        Y_true = Y[:, 1]
        logger.log_info("Generating classification report.")
        logger.log_info(classification_report(Y_true, Y_pred, digits=5))

        data_getter = DataGetter()
        deltas = data_getter.get_deltas()
        deltas = deltas[0:Y_pred.shape[0]]
        deltas["Y_Pred"] = Y_pred
        logger.log_info("Saving predictions to output file - ../output/df_predictions.xlsx")
        deltas.to_excel("../output/df_predictions.xlsx")

    def _log_evaluation_config(self, model_json, model_weights, X, Y):
        logger.log_info(
            "Using model architecture from - {model_arch_path} and weights from - {model_weights_path}.".format(
                model_arch_path=model_json, model_weights_path=model_weights))
        logger.log_info("Testing period from {start} to {end}.".format(start=config.TESTING_DATE_START,
                                                                       end=config.TESTING_DATE_END))

    def start_model_evaluation(self):
        logger.log_info("Starting model evaluation.")
        X, Y = self.get_testing_data()
        model_json, model_weights = config.get_model_json_and_weights_path()
        self._log_evaluation_config(model_json, model_weights, X, Y)
        self._evaluate_model(model_json, model_weights, X, Y)
