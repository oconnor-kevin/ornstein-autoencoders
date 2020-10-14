import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances

def make_mds_plot(codes, labels):
    codes = codes - np.mean(codes)
    codes = codes / (np.max(codes) - np.min(codes))
    similarities = euclidean_distances(codes)
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=0,
        dissimilarity="precomputed",
        n_jobs=1)
    pos = mds.fit(similarities).embedding_
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        c=labels,
        s=50,
        lw=0)
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.show()
