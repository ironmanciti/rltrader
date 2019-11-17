import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import sgd

class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(512, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.2))
        self.model.add(BatchNormalization())        # training 속도 향상
        self.model.add(LSTM(256, return_sequences=True, dropout=0.2))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(128, return_sequences=False, dropout=0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        #self.model.add(Activation('sigmoid'))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')
        # self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    def predict(self, sample):
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    # fit : 주어진 epoch 동안 dataset 을 iteration 하며 train
    # train_on_batch : data 의 single batch 에 대해 single gradient update 하여 training loss return
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)     # 정책신경망을 HDF5 file 로 저장

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
