"""An example for trianing & evaluating Multilayer perceptron."""

import ml


# Load the MNIST handwritten digits data.
# X : the input patterns (28x28=784 dimensional vectors).
# T : the label for each input pattern in X (represented in 1-of-K (a.k.a one-hot) coding scheme).
(X_train, T_train), (X_test, T_test) = ml.load_data('mnist')

# Create a new `mlp_classifier' object with
# the input layer with 784 units, hidden layers with 200, 50 units, resprectively,
# and the output layer with 10 units.
# The activation function for the hidden layers is ReLU, and
# the one for the output layer is automatically set to softmax ('cuz this is a multiclass classifier).
# In addition, the Dropout method is used in order to avoid overfit.
# By the setting below, 10% of input units and 30% of the hidden units are dropped out.
net = ml.nn.mlp_classifier.from_shape(
    [784, 200, 50, 10], hidden_act='ReLU',
    dropout_ratio=[0.1, 0.3, 0.3]
)

# Train the network.
# The first 10000 patterns in (X_train, T_train) are used for validation,
# and the rest (50000 patterns) are used for training.
# `eta0=1e-2` : The initial value of learning rate
# `lamb=5e-5` : The coeffiecient of weight decay
# `optimizer='Momentum` : the Momentum method is used in updating parameter
# `max_epoch=300' : Training continues for at most 300 epochs. but
# early stopping mode is ON by default, so it could be aborted before it reaches `max_epoch`.
# Not that training also can be stopped manually by `KeyboardInterrupt` (Ctrl+C).
net.train(
    X_train=X_train[10000:], T_train=T_train[10000:],
    X_val=X_train[:10000], T_val=T_train[:10000],
    eta0=1e-2, lamb=5e-5, optimizer='Momentum',
    max_epoch=300
)

# (uncomment this to replace the existing pickle file with a new one)
# net.save('pkl/best')

# Evaluation of the network's classification performance
print("(1) test for ml.nn.mlp_classifier's test() method")
print("Training ", end="")
net.test(X_train, T_train, verbose=True)
print("Test ", end="")
net.test(X_test, T_test, verbose=True)

print("(2) test for ml.evaluate")
print("Training data :")
print(ml.evaluate(net, X_train, T_train))
print("Test data : ")
print(ml.evaluate(net, X_test, T_test))
