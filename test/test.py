import numpy as np
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

net = ml.nn.mlp_classifier.from_shape(
    [784, 200, 50, 10], hidden_act='LeakyReLU',
    dropout_ratio=[0.1, 0.4, 0.25], inverted=True
)
net.train(
    X_train=X_train, T_train=T_train,
    X_val=X_test, T_val=T_test,
    eta0=1e-2, lamb=0.00005, optimizer='Momentum',
    max_epoch=3000, early_stopping=False
)

print("Train: ", end="")
net.test(X_train, T_train, verbose=True)
print("Test : ", end="")
net.test(X_test, T_test, verbose=True)
