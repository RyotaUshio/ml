"""モジュールのimportやデータの読み込みにより, interactive shellでのテストを効率化するためのスクリプト.
"""

import numpy as np
import matplotlib.pyplot as plt
import ml

# The original MNIST dataset
(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

# 3-D features extracted by MLP's hidden layers
X_train_3d = np.load('npy/mlp3d_train.npy')
X_test_3d = np.load('npy/mlp3d_test.npy')

