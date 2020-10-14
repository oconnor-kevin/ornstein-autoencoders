import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K

import data_utils as du
import model as oaemod
import discriminator as disc
import make_mds_plot as mds

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
    'n_samples':10,
    'lambda1':10.0,
    'lambda2':10.0,
    'ae_learning_rate':0.00001,
    'disc_learning_rate':0.00001
    #'ae_learning_rate':0.01,
    #'disc_learning_rate':0.005
}

def kernel(A, B):
    return tf.reduce_sum((tf.expand_dims(A, 1)-tf.expand_dims(B, 0))**2,2)

def oae_decoder_loss(data, reconstructions, discriminator_output):
    (b, z, x) = data
    (b_tilde, z_tilde, x_tilde) = reconstructions
    n = model_params['n_subjects']
    rec_loss = tf.reduce_mean(
        tf.reduce_sum(tf.math.square(x - x_tilde), -1))
    disc_loss = tf.reduce_mean(tf.math.log(discriminator_output))
    kernel_loss = tf.reduce_sum(-(kernel(b, b) + kernel(b_tilde, b_tilde))/(n*(n-1.0)) + 2.0*kernel(b, b_tilde)/(n**2.0))
    return rec_loss - model_params['lambda1']*disc_loss + model_params['lambda2']*kernel_loss

def oae_discriminator_loss(f_z, f_z_tilde):
    return tf.reduce_sum(tf.math.log(f_z) + tf.math.log(1 - f_z_tilde))

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "\n" if iteration < total else "\n\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end = end)


oae_mod = oaemod.OAE(model_params)
disc_mod = disc.Discriminator(model_params)
ae_optimizer = keras.optimizers.Adam(lr=model_params['ae_learning_rate'])
disc_optimizer = keras.optimizers.Adam(lr=model_params['disc_learning_rate'])
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
        with tf.GradientTape(persistent=True) as tape:
            (encoded, nu, mu, sigma) = oae_mod.encode(batch, labels)
            decoded = oae_mod.decode(encoded, batch)
            (b, z) = oae_mod.sample_prior()
            disc_code = disc_mod.call(encoded[1])
            disc_prior = disc_mod.call(z)
            # VAE loss
            ae_loss = tf.reduce_mean(
                oae_decoder_loss((b, z, batch), encoded + (decoded,), disc_code)
            )
            # Discriminator loss
            disc_loss = tf.reduce_mean(
                -oae_discriminator_loss(disc_prior, disc_code)
            )
        # Gradient step for autoencoder
        gradients = tape.gradient(ae_loss, oae_mod.trainable_variables)
        ae_optimizer.apply_gradients(zip(gradients, oae_mod.trainable_variables))
        mean_loss(ae_loss)
        # Gradient step for disciminator
        gradients = tape.gradient(disc_loss, disc_mod.trainable_variables)
        disc_optimizer.apply_gradients(zip(gradients, disc_mod.trainable_variables))
        mean_loss(disc_loss)
        print_status_bar(step * model_params['n_subjects'] * model_params['n_samples'], n_train, mean_loss, metrics)
    print_status_bar(n_train, n_train, mean_loss)
    for metric in [mean_loss] + metrics:
        metric.reset_states()

# Plot codes
(batch, labels) = du.random_batch(
    test_data,
    n_subjects=model_params['n_subjects'],
    n_samples=model_params['n_samples'])
(codes, nu, mu, sigma) = oae_mod.encode(batch, labels)
mu = tf.reshape(mu, [-1, model_params['hidden_dim']])
mds.make_mds_plot(mu, labels)


# Get number of parameters
trainable_count = np.sum([K.count_params(w) for w in oae_mod.trainable_weights])
print('AE trainable params: {:,}'.format(trainable_count))
trainable_count = np.sum([K.count_params(w) for w in disc_mod.trainable_weights])
print('Discriminator trainable params: {:,}'.format(trainable_count))
