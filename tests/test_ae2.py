"""test_ae.pyで得られたオートエンコーダにより次元圧縮をおこない、それを用いてMLPを学習させてみる"""

import numpy as np
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

ae = ml.load('pkl/ae')
ae.forward_prop(X_train)
X_train_encoded = ae[3].z.copy()
ae.forward_prop(X_test)
X_test_encoded = ae[3].z.copy()

net = ml.nn.mlp_classifier.from_shape([32, 20, 10])

net.train(
    X_train=X_train_encoded[:-6000], T_train=T_train[:-6000],
    X_val=X_train_encoded[-6000:], T_val=T_train[-6000:],
    eta0=1e-2, optimizer='SGD',
    tol=0
)

