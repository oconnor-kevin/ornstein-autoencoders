import tensorflow as tf
import numpy as np
from tensorflow import keras

class BasicAE(keras.Model):
    def __init__(self, model_params, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        ## Mu
        self.mu_hidden1 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu", input_shape=self.model_params['input_shape'])
        self.mu_hidden2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu")
        self.mu_dense1 = keras.layers.Dense(self.model_params['hidden_dim'])
        ## Sigma
        self.log_sigma_hidden1 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu")
        self.log_sigma_hidden2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu")
        self.log_sigma_dense1 = keras.layers.Dense(self.model_params['hidden_dim'])
        ## Decoding
        self.decode_dense1 = keras.layers.Dense(64*7*7)
        self.hidden2 = keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu")
        self.hidden3 = keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="selu")

    def sampling(self, n, mu, sigma):
        epsilon = tf.random.normal(
            shape = (n, self.model_params['hidden_dim']),
            mean = 0.0,
            stddev = self.model_params['epsilon_std']
        )
        return mu + sigma * epsilon

    def encode(self, inputs):
        # Encoder
        #expand = keras.layers.Reshape(inputs.shape+(1,), dtype=tf.float64)(inputs)
        expand = tf.expand_dims(inputs, -1)
        ## Mu
        mu = self.mu_hidden1(expand)
        mu = keras.layers.MaxPool2D(pool_size=2)(mu)
        mu = self.mu_hidden2(mu)
        mu = keras.layers.MaxPool2D(pool_size=2)(mu)
        mu = keras.layers.Flatten()(mu)
        mu = self.mu_dense1(mu)
        ## Sigma
        log_sigma = self.log_sigma_hidden1(expand)
        log_sigma = keras.layers.MaxPool2D(pool_size=2)(log_sigma)
        log_sigma = self.log_sigma_hidden2(log_sigma)
        log_sigma = keras.layers.MaxPool2D(pool_size=2)(log_sigma)
        log_sigma = keras.layers.Flatten()(log_sigma)
        log_sigma = self.log_sigma_dense1(log_sigma)
        sigma = tf.exp(log_sigma)
        ## Get codes
        codes = self.sampling(inputs.shape[0], mu, sigma)
        return (codes, mu, sigma)

    def decode(self, codes, inputs):
        # Decoder
        decoded = self.decode_dense1(codes)
        decoded = keras.layers.Reshape([7, 7, 64])(decoded)
        decoded = self.hidden2(decoded)
        decoded = self.hidden3(decoded)
        decoded = keras.layers.Reshape(inputs.shape[1:])(decoded)
        return decoded
