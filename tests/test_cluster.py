"""An example script for testing the clustering algorithms in ml.
"""

import numpy as np
import ml

# The original MNIST dataset
(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

# 3-D features extracted by MLP's hidden layers
X_train_3d = np.load('npy/mlp3d_train.npy')
X_test_3d = np.load('npy/mlp3d_test.npy')

evals = dict()


# _________________________________________________________________

# K-means
kmeans = ml.cluster.k_means(X_train_3d[:2000], k=10, plot=False)
kmeans.scatter()
evals['k_means'] = ml.evaluate(kmeans)

# Competitive learning
clnet = ml.cluster.competitive_net(
    X_train_3d[:2000], k=10,
    optimizer='AdaGrad', eta0=1e-2, tol=1e-5,
    max_epoch=10
)
clnet.scatter()
evals['competitive_net'] = ml.evaluate(clnet)

# EM algorithm
em = ml.cluster.em(X_train_3d[:2000], k=10)
em.scatter()
evals['em'] = ml.evaluate(em)


# _________________________________________________________________

print("\n" + "-------------------"*5)
for v in evals.values():
    print(v)
    print("-------------------"*5)
