import sys
import os
sys.path.append(os.pardir)
import ml

(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')
hidden_shape = [int(arg) for arg in sys.argv[1:]]

net = ml.classify.mlp_classifier.from_shape(
    shape=[784] + hidden_shape + [10], hidden_act='tanh'
)

print(net)

net.train(
    X_train, T_train, batch_size=1,
    early_stopping=False, max_epoch=1000, how='stdout'
)

net.save(
    os.path.abspath(ml.__path__[0]) + f'/../pkl/hidden_shape_' + '_'.join(sys.argv[1:]) + '_online'
)
