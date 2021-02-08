"""An example script for testing the clustering algorithms in ml.
"""
import ml
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools

# doesn't work

# The original MNIST dataset
# (X_train, T_train), (X_test, T_test) = ml.load_data('mnist')
# net = ml.load('pkl/mlp3d_dropout_weight_decay.pkl')
# net.forward_prop(X_train)
# X = net[-2].z.copy()

# em = ml.cluster.em(X, k=10)
# em.scatter()
# print(ml.evaluate(em))

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
km = ml.cluster.k_means(*args, plot=False)
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
after_scatter(f'K-means: {accuracy(km) * 100:.4f}%')
em.scatter()
after_scatter(f'EM algorithm: {accuracy(em) * 100:.4f}%')
ml.utils.scatter(args[0], T=labels)
after_scatter('Original')
plt.show()
