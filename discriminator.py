import tensorflow as tf
import numpy as np
from tensorflow import keras

class Discriminator(keras.Model):
    def __init__(self, model_params, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.disc_dense1 = keras.layers.Dense(256, input_shape=(self.model_params['hidden_dim'],))
        self.disc_dense2 = keras.layers.Dense(512)
        self.disc_dense3 = keras.layers.Dense(128)
        self.disc_output = keras.layers.Dense(1, activation='sigmoid')

    def call(self, codes):
        codes = self.disc_dense1(codes[1])
        codes = keras.layers.Dropout(0.2)(codes)
        codes = keras.layers.BatchNormalization()(codes)
        codes = self.disc_dense2(codes)
        codes = keras.layers.Dropout(0.2)(codes)
        codes = keras.layers.BatchNormalization()(codes)
        codes = self.disc_dense3(codes)
        codes = keras.layers.Dropout(0.2)(codes)
        codes = keras.layers.BatchNormalization()(codes)
        disc_out = self.disc_output(codes)
        return disc_out
