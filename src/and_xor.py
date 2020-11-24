import numpy as np
from neuron import *

alpha = 100
debug_view = (
    True
    #False
)

# # x0=1を明示的に指定する場合

# ## ANDの学習
# training_dataset = (
#     DataSet(
#         np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]), 
#         np.array([0, 0, 0, 1]),
#         True
#         )
#     )
# w0 = np.random.rand(3)
# nr = Neuron(w0, thd)
# Simple(nr, training_dataset)
# test(nr, training_dataset)

# # XORは線形分離不可能なので次元を上げてやらないとできない
# training_dataset = (
#     DataSet(
#         np.array([[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]), 
#         np.array([0, 1, 1, 0]),
#         True
#     )
# )
# w0 = np.random.rand(4)
# nr = Neuron(w0, thd)
# Simple(nr, training_dataset, debug_view=debug_view)
# test(nr, training_dataset)


# x0 = 1を自動で
## ANDの学習
training_dataset = (
    DataSet(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
        np.array([0, 0, 0, 1]),
        )
    )
w0 = np.random.rand(3)
nr = Neuron(w0, thd)
Simple(nr, training_dataset)
test(nr, training_dataset)

# XORは線形分離不可能なので次元を上げてやらないとできない
training_dataset = (
    DataSet(
        np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]), 
        np.array([0, 1, 1, 0])
    )
)
w0 = np.random.rand(4)
nr = Neuron(w0, thd)
Simple(nr, training_dataset, debug_view=debug_view)
test(nr, training_dataset)


### SGD ###

def sigmoid(u, alpha=alpha, threshold=0.001):
    y = (np.tanh(alpha*u/2.0) + 1.0) / 2.0
    ## 桁落ち防止!!
    y = y if y < 1 - threshold else 1 - threshold
    y = y if y > threshold else threshold
    return y

## andの学習
training_dataset = (
    DataSet(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
        np.array([0, 0, 0, 1]),
        )
    )
w0 = np.random.rand(3)
nr = Neuron(w0, sigmoid)
SGD(nr, training_dataset, debug_view=debug_view, alpha=alpha)
test(nr, training_dataset)

# XORは線形分離不可能なので次元を上げてやらないとできない
training_dataset = (
    DataSet(
        np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]), 
        np.array([0, 1, 1, 0])
    )
)
w0 = np.random.rand(4)
nr = Neuron(w0, sigmoid)
SGD(nr, training_dataset, debug_view=debug_view, alpha=alpha)
test(nr, training_dataset)
