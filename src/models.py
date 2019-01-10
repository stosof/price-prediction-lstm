import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from data_getter import DataGetter
import config


class LSTM_NN(object):
    def __init__(self):
        self.X = None

    def get_compiled_lstm(self):
        model = Sequential()
        model.add(LSTM(1024, input_shape=(config.SEQUENCE_LENGTH, 156)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_training_data(self):
        X = self._get_training_data_x()
        Y = self._get_training_data_y()
        return X, Y

    def _get_training_data_x(self):
        data_getter = DataGetter()
        reshaped_data_lstm = data_getter.get_reshaped_data_for_lstm()
        self.X = reshaped_data_lstm
        return reshaped_data_lstm

    def _get_training_data_y(self):
        data_getter = DataGetter()
        df_result = data_getter.get_deltas()
        df_result = df_result[config.TRAINING_DATA_TARGET]
        y = df_result.values
        if self.X is None:
            raise Exception('X needs to be defined before defining Y. Run _get_training_data_x before this method.')
        y = y[0:self.X.shape[0]]
        y = self._one_hot_encode(y, 2)
        return y

    def _one_hot_encode(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def fit_model(self, X, Y, n_batch, nb_epoch):
        model = self.get_compiled_lstm()
        callbacks = self._get_callbacks()
        history = model.fit(X, Y, epochs=nb_epoch, batch_size=n_batch, verbose=1, shuffle=False, callbacks=callbacks,
                  validation_data=(X, Y))
        return history

    def _get_callbacks(self):
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list

    def start_model_training(self):
        X, Y = self.get_training_data()
        self.fit_model(X=X, Y=Y, n_batch=config.TRAINING_BATCH_SIZE, nb_epoch=config.TRAINING_EPOCHS)

if __name__ == "__main__":
    lstm = LSTM_NN()
    lstm.start_model_training()
