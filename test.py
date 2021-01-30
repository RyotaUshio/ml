import nn, utils
import numpy as np

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')

net = nn.mlp_classifier.from_shape([784, 500, 200, 10], dropout_ratio=[0.1, 0.4, 0.4])
net.train(X_train=X_train, T_train=T_train,
          X_val=X_test, T_val=T_test,
          eta0=1e-2, lamb=0.0001, optimizer='Momentum',
          max_epoch=830, 
          early_stopping=False)

print("Train: ", end="")
net.test(X_train, T_train, log=False, verbose=True)
print("Test : ", end="")
net.test(X_test, T_test, verbose=True)
