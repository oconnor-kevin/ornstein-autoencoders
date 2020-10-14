import tensorflow as tf
import numpy as np
from tensorflow import keras

class OAE(keras.Model):
    def __init__(self, model_params, **kwargs):
        super().__init__(**kwargs)
        self.model_params = model_params
        ## Nu
        self.nu_hidden1 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="selu", input_shape=(self.model_params['n_samples'],) + self.model_params['input_shape'] + (1,))
        self.nu_hidden2 = keras.layers.Conv2D(512, kernel_size=3, padding="same", activation="selu")
        self.nu_dense1 = keras.layers.Dense(self.model_params['hidden_dim'])
        self.nu_dense2 = keras.layers.Dense(self.model_params['hidden_dim'])
        ## Mu
        self.mu_hidden1 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="selu", input_shape=self.model_params['input_shape'])
        self.mu_hidden2 = keras.layers.Conv2D(512, kernel_size=3, padding="same", activation="selu")
        self.mu_dense1 = keras.layers.Dense(self.model_params['hidden_dim'])
        ## Sigma
        self.log_sigma_hidden1 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="selu")
        self.log_sigma_hidden2 = keras.layers.Conv2D(512, kernel_size=3, padding="same", activation="selu")
        self.log_sigma_dense1 = keras.layers.Dense(self.model_params['hidden_dim'])
        ## Decoding
        self.decode_dense1 = keras.layers.Dense(512*7*7)
        self.hidden2 = keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", activation="selu")
        self.hidden3 = keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="selu")

    def sample_prior(self):
        # Draw b's.
        b = tf.random.normal(
            shape = (self.model_params['n_subjects'], self.model_params['hidden_dim']),
            mean = 0.0,
            stddev = self.model_params['tau0']
        )
        b_tiled = tf.tile(tf.expand_dims(b, 1), [1, self.model_params['n_samples'], 1])
        # Draw z's.
        z = tf.random.normal(
            shape = (self.model_params['n_subjects'],
                     self.model_params['n_samples'],
                     self.model_params['hidden_dim']),
            mean = self.model_params['mu0'],
            stddev = self.model_params['sigma0']
        ) + b_tiled

        return (b, z)

    def sample_posterior(self, nu, mu, sigma):
        # Draw b's.
        b = tf.random.normal(
            shape = (self.model_params['n_subjects'],
                     self.model_params['hidden_dim']),
            stddev = self.model_params['tau']
        ) + nu
        b_tiled = tf.tile(tf.expand_dims(b, 1), [1, self.model_params['n_samples'], 1])
        # Draw z's.
        z = tf.random.normal(
            shape = (self.model_params['n_subjects'],
                     self.model_params['n_samples'],
                     self.model_params['hidden_dim'])
        )*sigma + mu + b_tiled
        return (b, z)

    def encode(self, inputs, labels):
        # Encoder
        expand = tf.expand_dims(inputs, -1)
        ## Nu
        # Group data by labels.
        labels = tf.cast(labels, tf.float32)
        inputs_sorted = tf.gather(expand, tf.argsort(labels))
        #inputs_grouped = tf.stack(tf.split(inputs_sorted, self.model_params['n_subjects']))
        #nu = self.nu_hidden1(inputs_grouped)
        nu = self.nu_hidden1(inputs_sorted)
        nu = keras.layers.MaxPool2D(pool_size=2)(nu)
        nu = self.nu_hidden2(nu)
        nu = keras.layers.MaxPool2D(pool_size=2)(nu)
        nu = keras.layers.Flatten()(nu)
        nu = self.nu_dense1(nu)
        nu_grouped = tf.reshape(
            tf.stack(tf.split(nu, self.model_params['n_subjects'])),
            [self.model_params['n_subjects'], -1])
        nu = self.nu_dense2(nu_grouped)
        ## Mu
        mu = self.mu_hidden1(expand)
        mu = keras.layers.MaxPool2D(pool_size=2)(mu)
        mu = self.mu_hidden2(mu)
        mu = keras.layers.MaxPool2D(pool_size=2)(mu)
        mu = keras.layers.Flatten()(mu)
        mu = self.mu_dense1(mu)
        mu = tf.reshape(mu, [self.model_params['n_subjects'], self.model_params['n_samples'], -1])
        ## Sigma
        log_sigma = self.log_sigma_hidden1(expand)
        log_sigma = keras.layers.MaxPool2D(pool_size=2)(log_sigma)
        log_sigma = self.log_sigma_hidden2(log_sigma)
        log_sigma = keras.layers.MaxPool2D(pool_size=2)(log_sigma)
        log_sigma = keras.layers.Flatten()(log_sigma)
        log_sigma = self.log_sigma_dense1(log_sigma)
        sigma = tf.reshape(tf.exp(log_sigma), [self.model_params['n_subjects'], self.model_params['n_samples'], -1])
        ## Get codes
        codes = self.sample_posterior(nu, mu, sigma)
        return (codes, nu, mu, sigma)

    def decode(self, codes, inputs):
        codes = tf.reshape(codes[1], [-1, self.model_params['hidden_dim']])
        decoded = self.decode_dense1(codes)
        decoded = keras.layers.Reshape([7, 7, -1])(decoded)
        decoded = self.hidden2(decoded)
        decoded = self.hidden3(decoded)
        decoded = keras.layers.Reshape(inputs.shape[1:])(decoded)
        return decoded
