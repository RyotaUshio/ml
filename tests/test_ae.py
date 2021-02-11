import numpy as np
import matplotlib.pyplot as plt

import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

ae = ml.feature.autoencoder.fit(
    X_train, [128, 64, 32], 
    encode_act='ReLU', decode_act=['sigmoid', 'ReLU', 'ReLU'], 
    X_val=X_test,
    eta0=0.0001,
    optimizer='Momentum',
    lamb=0.0001,
    max_epoch=500,
    patience_epoch=30
)

ae.save('pkl/ae')
ml.imshow(ae(X_train[0]).flatten())
plt.pause(1)
ml.imshow(ae(X_test[0]).flatten())
plt.pause(1)
