import numpy as np
import matplotlib.pyplot as plt

import nn, utils

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')

net = nn.mlp_classifier.fit(
    X_train, T_train,
    hidden_shape=[500, 200, 3],
    hidden_act=['ReLU', 'ReLU', 'tanh'],
    X_val=X_test, T_val=T_test, 
    eta0=1e-2, lamb=0.00005, optimizer='Momentum',
    max_epoch=300
)

net(X_train)
labels = utils.vec2label(T_train)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    x, y, z = net[-2].z[labels == i].T
    ax.scatter(x, y, z, label=f'{i}')

fig.legend()

