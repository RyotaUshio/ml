import sys
import os
sys.path.append(os.pardir)
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')
n_hidden_unit = int(sys.argv[1])

net = ml.classify.mlp_classifier.from_shape(
    shape=[784, n_hidden_unit, 10]
)
net.train(
    X_train, T_train, early_stopping=False
)

net.save(os.path.abspath(ml.__path__[0]) + f'/../pkl/n_hidden_unit_{n_hidden_unit}')
