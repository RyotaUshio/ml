import numpy as np
import utils
import classify
import evaluate as ev

(X_train, T_train), (X_test, T_test) = utils.load_data('mnist')

# 単純パーセプトロンで扱えるように、0と1のデータだけ取り出す
T_train = utils.digit(T_train)
T_test = utils.digit(T_test)

X_train = X_train[T_train <= 1]
X_test = X_test[T_test <= 1]
T_train = (T_train[T_train <= 1]).reshape(-1, 1)
T_test = (T_test[T_test <= 1]).reshape(-1, 1)

# np.save('mnist_binary_X_train.npy', X_train)
# np.save('mnist_binary_X_test.npy', X_test)
# np.save('mnist_binary_T_train.npy', T_train)
# np.save('mnist_binary_T_test.npy', T_test)

pctr = classify.simple_perceptron.fit(X_train, T_train)

pctr.test(X_test, T_test, True)

# pctr.save('pkl/perceptron.pkl') -> utils.load('pkl/perceptron.pkl')で読み込める.
