import tensorflow as tf
import numpy as np
from tensorflow import keras

import data_utils as du
import basic_ae_model as basicmod
import make_mds_plot as mds
import make_reconstructions_plot as recplt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Model parameters
model_params = {
    'hidden_dim':8,
    'tau0':0.02,
    'mu0':0.0,
    'tau':10.0,
    'epsilon_std':1.0,
    'sigma0':1.0,
    'input_shape':(28, 28),
    'n_epochs':2,
    'n_subjects':10,
    'n_samples':100,
    'lambda1':10.0,
    'lambda2':10.0,
    'learning_rate':0.001
}

def vae_loss(data, reconstructions, mu, sigma):
    rec_loss = keras.losses.binary_crossentropy(tf.reshape(data, (data.shape[0], -1)),
        tf.reshape(reconstructions, (reconstructions.shape[0], -1)))
    kl_loss = -0.5*tf.reduce_sum(1 + 2.0*tf.math.log(sigma) - tf.math.pow(sigma, 2.0) - tf.math.pow(mu, 2.0), -1)
    return rec_loss + kl_loss / np.prod(model_params['input_shape'])

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "\n" if iteration < total else "\n\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end = end)


basic_mod = basicmod.BasicAE(model_params)
optimizer = keras.optimizers.Nadam(lr=model_params['learning_rate'])
mean_loss = keras.metrics.Mean()
metrics = []

# Training
(train_data, test_data) = du.load_data()
n_train = np.sum(np.array([train_data[k].shape[0] for k in train_data.keys()]))
n_steps = np.ceil(n_train / (model_params['n_subjects'] * model_params['n_samples'])).astype('int') # TODO: automate
for epoch in range(1, model_params['n_epochs'] + 1):
    print("Epoch {}/{}".format(epoch, model_params['n_epochs']))
    for step in range(1, n_steps + 1):
        (batch, labels) = du.random_batch(
            train_data,
            n_subjects=model_params['n_subjects'],
            n_samples=model_params['n_samples'])
        with tf.GradientTape() as tape:
            (encoded, mu, sigma) = basic_mod.encode(batch)
            decoded = basic_mod.decode(encoded, batch)
            main_loss = tf.reduce_mean(vae_loss(batch, decoded, mu, sigma))
            loss = tf.add_n([main_loss] + basic_mod.losses)
        gradients = tape.gradient(loss, basic_mod.trainable_variables)
        optimizer.apply_gradients(zip(gradients, basic_mod.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(batch, decoded)
        print_status_bar(step * model_params['n_subjects'] * model_params['n_samples'], n_train, mean_loss, metrics)
    print_status_bar(n_train, n_train, mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()

# Plot codes
(batch, labels) = du.random_batch(
    train_data,
    n_subjects=10,
    n_samples=100)
(codes, mu, sigma) = basic_mod.encode(batch)
mds.make_mds_plot(mu, labels)
