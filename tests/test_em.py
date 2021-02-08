"""An example script for testing the clustering algorithms in ml.
"""
import ml
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

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

def scatter(cluster):
    cluster.scatter()
    plt.gca().set_aspect('equal')


X = [
    gen([-5, 0], 8, 1000),
    gen([6, 3], 1, 1000),
    gen([6, -3], 1, 1000)
]
labels = np.array([[i for _ in X[i]] for i in range(len(X))]).ravel()

args = [np.vstack(X), len(X)]
km = ml.cluster.k_means(*args, plot=False)
em = ml.cluster.em(*args)

scatter(km)
plt.title('K-means')
scatter(em)
plt.title('EM algorithm')

ml.utils.scatter(args[0], T=labels)
plt.title('Original')
