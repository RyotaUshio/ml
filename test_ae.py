import numpy as np
import matplotlib.pyplot as plt

import nn, utils, feature as ft

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')

ae = ft.autoencoder.fit(
    X_train, [128, 64, 32], 
    encode_act='ReLU', decode_act=['sigmoid', 'ReLU', 'ReLU'], 
    X_val=X_test,
    lamb=0,
    optimizer='Momentum'
)


ae(X_train)
labels = utils.vec2label(T_train)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(10):
    x, y, z = ae[3].z[labels == i].T
    ax.scatter(x, y, z, label=f'{i}')

fig.legend()


