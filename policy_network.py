import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import sgd

class PolicyNetwork:

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)     # 정책신경망을 HDF5 file 로 저장

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
