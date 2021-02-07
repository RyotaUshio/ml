"""主成分分析によりMNISTデータを784次元から3次元に圧縮し、可視化する。
"""

import numpy as np
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

pca = ml.feature.pca(X_train)
X_train_reduced = pca.reduce(3) # 寄与率の大きい方から3次元分をとる

ml.scatter(X_train_reduced, T_train)
