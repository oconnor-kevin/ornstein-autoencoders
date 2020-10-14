import tensorflow as tf
import numpy as np
from tensorflow import keras


def load_data(dataset='mnist'):
    if dataset is 'mnist':
        data = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = data.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_data, test_data = {}, {}
        for digit in np.unique(y_train):
            idxs = np.where(y_train == digit)
            train_data[digit] = np.squeeze(x_train[idxs,:,:])
        for digit in np.unique(y_test):
            idxs = np.where(y_test == digit)
            test_data[digit] = np.squeeze(x_test[idxs,:,:])
    else:
        raise NotImplementedError
    return (train_data, test_data)

def random_batch(X, n_subjects=5, n_samples=100):
    subject_ids = np.unique(list(X.keys()))
    # Select subjects
    subjects = np.random.choice(subject_ids, min(n_subjects, len(subject_ids)), replace=False)
    # Select samples from each subject
    samples = []
    labels = []
    for subject in subjects:
        sample_idxs = np.random.choice(X[subject].shape[0], n_samples)
        samples.append(np.take(X[subject], sample_idxs, axis=0))
        labels.append([subject for _ in np.arange(len(sample_idxs))])
    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels)
    # Shuffle samples
    permutation = np.random.permutation(samples.shape[0])
    samples = samples[permutation]
    labels = labels[permutation]
    return((samples, labels))
