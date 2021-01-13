import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dataclasses
import doctest
from typing import Sequence, List, Callable
from importlib import reload
import nn
reload(nn)

if first:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # 2クラス分類として
    
    T = np.array([[0, 1, 1, 0]]).T

    act_funcs = [
        nn.linear(),
        nn.sigmoid(),
        nn.sigmoid()
    ]
    num = [2, 2, 1]
    net = nn.mlp.from_num(num=num, act_funcs=act_funcs, loss=nn.cross_entropy(), sigma=2.0)

    first=False

def two_class():

    net.train(X, T, 
              eta=0.1,
              max_iter=100000,
              log_cond=lambda m, i: m%1000==0 and i==0
    )
    net.test(X, T)

# プロット
def plot(margin=0.5, size=20):
    x_ = y_ = np.linspace(-margin, 1+margin, 100)
    z = []
    for x in x_:
        for y in y_:
            net.forward_prop(np.array([x, y]))
            z.append(net[-1].z)
    fig, ax = plt.subplots()

    x, y = np.meshgrid(x_, y_)
    z = np.array(z).reshape(x.shape)
    ax.contour(x, y, z, [0.5])
    ax.set_aspect('equal')
    ax.scatter(X.T[0], X.T[1], s=size, c='k')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(linestyle="--")

    return fig, ax


# 多クラス分類として
def mul_class():
    T = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    act_funcs = [
        nn.linear(),
        nn.sigmoid(),
        nn.softmax()
    ]
    num = [2, 2, 2]
    net = nn.mlp.from_num(num=num, act_funcs=act_funcs, sigma=1.0)

    net.train(X, T, 
              eta=0.005,
              max_iter=1000000,
              log_cond=lambda m, i: m%1000==0
    )
    net.test(X, T)
