"""An example script for testing the clustering algorithms in ml.
"""
import ml
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools

def gen(mean, cov, size):
    if isinstance(cov, (float, int)):
        cov = np.identity(len(mean)) * cov
    return scipy.stats.multivariate_normal.rvs(
        mean=mean, cov=cov, size=size
    )

def after_scatter(title):
    plt.gca().set_aspect('equal')
    plt.title(title)

X = [
    gen([-2, 0], [[8, 0], [0, 24]], 2000),
    gen([8, 3], [[3, 0], [0, 0.5]], 6000),
    gen([6, -4], [[0.5, 0], [0, 4]], 1000)
]
labels = np.concatenate([[i for _ in X[i]] for i in range(len(X))])
args = [np.vstack(X), len(X)]

# K-means
km = ml.cluster.k_means(*args, plot=False)
# Competitive Learning
cl = ml.cluster.competitive_net(*args, max_epoch=30)
# EM algorithm
em = ml.cluster.em(*args, least_improve=1e-3)

def accuracy(cluster):
    best = 0
    for order in itertools.permutations(range(len(X))):
        u, i = np.unique(labels, return_inverse=True)
        n_ok = np.count_nonzero(cluster.labels == np.array(order)[i])
        if n_ok > best:
            best = n_ok
    return best / len(labels)

km.scatter()
after_scatter(f'K-means: {accuracy(km) * 100:.2f}%')
cl.scatter()
after_scatter(f'Competitive Learning: {accuracy(km) * 100:.2f}%')
em.scatter(contours=True)
after_scatter(f'EM algorithm: {accuracy(em) * 100:.2f}%')
ml.utils.scatter(args[0], T=labels)
after_scatter('Original')

plt.show()
