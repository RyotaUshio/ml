import numpy as np
import sys, os
sys.path.append(os.pardir)
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

# 単純パーセプトロンで扱えるように、0と1のデータだけ取り出す
T_train = ml.utils.digit(T_train)
T_test = ml.utils.digit(T_test)

X_train = X_train[T_train <= 1]
X_test = X_test[T_test <= 1]
T_train = (T_train[T_train <= 1]).reshape(-1, 1)
T_test = (T_test[T_test <= 1]).reshape(-1, 1)

# np.save('mnist_binary_X_train.npy', X_train)
# np.save('mnist_binary_X_test.npy', X_test)
# np.save('mnist_binary_T_train.npy', T_train)
# np.save('mnist_binary_T_test.npy', T_test)

pctr = ml.classify.simple_perceptron.fit(X_train, T_train)

pctr.test(X_test, T_test, True)

# uncomment this to overwrite
pctr.save('pkl/perceptron.pkl') # -> ml.load('pkl/perceptron.pkl')で読み込める.
