import numpy as np
import utils, classify, cluster, feature as ft, evaluate as ev

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')
X_train_3d = np.load('npy/mlp3d_train.npy')
X_test_3d = np.load('npy/mlp3d_test.npy')

evals = dict()

# ____________________________ cluster ____________________________

# K-means
kmeans = cluster.k_means(X_train_3d[:2000], k=10)
evals['k_means'] = ev.evaluate(kmeans)

# Competitive learning
clnet = cluster.competitive_net(
    X_train_3d[:2000], k=10,
    optimizer='AdaGrad', eta0=1e-2, tol=1e-5,
    max_epoch=50
)
evals['competitive_net'] = ev.evaluate(clnet)

# EM algorithm
pass


# ____________________________ classify ____________________________

# K-nearest neighbors
knn = classify.k_nearest(X_train_3d, T_train, k=5)
knn.test(X_test_3d[:100], T_test[:100], verbose=True)
evals['k_nearest'] = ev.evaluate(knn, X_test_3d[:1000], T_test[:1000])

# Generative model
gen = classify.generative(X_train_3d, T_train)
gen.test(X_test_3d[:100], T_test[:100], verbose=True)
evals['generative'] = ev.evaluate(
    gen, X_test_3d[:1000], T_test[:1000],
    cov_identity=True, prior_equal=True
)

# Simple perceptron
bin_X_test  = np.load('npy/mnist_binary_X_test.npy')
bin_T_test  = np.load('npy/mnist_binary_T_test.npy')
pctr = utils.load('pkl/perceptron.pkl') # test_perceptron.pyで訓練
pctr.test(bin_X_test, bin_T_test, True)
evals['simple_perceptron'] = ev.evaluate(pctr, bin_X_test, bin_T_test)

# MLP
# 訓練時に使ったスクリプトは消失してしまったが、net.logにほとんどの情報は保存されている
net = utils.load('pkl/9887_ido.pkl')    
net.test(X_test, T_test, verbose=True)
evals['mlp_classifier'] = ev.evaluate(net, X_test, T_test)


# _________________________________________________________________

print("\n" + "-------------------"*5)
for v in evals.values():
    print(v)
    print("-------------------"*5)
