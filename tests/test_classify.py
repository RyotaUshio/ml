"""An example script for testing the classification algorithms in ml.
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

# K-nearest neighbors
knn = ml.classify.k_nearest(X_train_3d, T_train, k=5)
print('k_nearest ', end='')
knn.test(X_test_3d[:100], T_test[:100], verbose=True)
evals['k_nearest'] = ml.evaluate(knn, X_test_3d[:1000], T_test[:1000])

# Generative model
gen = ml.classify.generative(X_train_3d, T_train)
print('generative (no assumption of equality of the covariance matrices & prior probabilities) ', end='')
gen.test(X_test_3d[:100], T_test[:100], verbose=True)
evals['generative'] = ml.evaluate(
    gen, X_test_3d[:1000], T_test[:1000],
    cov_identity=True, prior_equal=True
)

# Simple perceptron
bin_X_test  = np.load('npy/mnist_binary_X_test.npy')
bin_T_test  = np.load('npy/mnist_binary_T_test.npy')
pctr = ml.load('pkl/perceptron.pkl') # test_perceptron.pyで訓練
pctr.test(bin_X_test, bin_T_test, True)
evals['simple_perceptron'] = ml.evaluate(pctr, bin_X_test, bin_T_test)

# MLP
# 訓練に時間がかかるのでpickle化したものを読み込んでいる
net = ml.load('pkl/best.pkl')    
net.test(X_test, T_test, verbose=True)
evals['mlp_classifier'] = ml.evaluate(net, X_test, T_test)


# _________________________________________________________________

print("\n" + "-------------------"*5)
for v in evals.values():
    print(v)
    print("-------------------"*5)
