"""MLPの訓練の例。"""

import numpy as np
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

net = ml.nn.mlp_classifier.from_shape(
    [784, 200, 50, 10], hidden_act='ReLU',
    dropout_ratio=[0.1, 0.3, 0.1], inverted=True
)
net.train(
    X_train=X_train[10000:], T_train=T_train[10000:],
    X_val=X_test[:10000], T_val=T_train[:10000],
    eta0=1e-2, lamb=5e-05, optimizer='Momentum',
    max_epoch=3000, early_stopping=False
)

net.save('pkl/best')

print("Train: ", end="")
net.test(X_train, T_train, verbose=True)
print("Test : ", end="")
net.test(X_test, T_test, verbose=True)
