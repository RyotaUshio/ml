import numpy as np
import matplotlib.pyplot as plt

import utils, feature as ft

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')

ae = ft.autoencoder.fit(
    X_train, [128, 64, 32], 
    encode_act='ReLU', decode_act=['sigmoid', 'ReLU', 'ReLU'], 
    X_val=X_test,
    lamb=0.0001,eta0=0.0001, 
    optimizer='Momentum', max_epoch=500
)

utils.imshow(ae(X_train[0]).ravel())
utils.imshow(ae(X_test[0]).ravel())
