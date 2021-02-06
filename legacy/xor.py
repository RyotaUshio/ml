import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dataclasses
from typing import Sequence, List, Callable
from importlib import reload
import nn
reload(nn)


first = True # インタプリタから再読込するときはコメントアウトする

if first:
    # データセット(真理値表)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])    
    T = np.array([[0, 1, 1, 0]]).T

    # 各層の活性化関数
    act_funcs = [
        None,
        'sigmoid',
        'sigmoid'
    ]
    # 各層のニューロン数
    num = [2, 2, 1]
    # mlpオブジェクトを生成
    net = nn.mlp.from_shape(shape=num, act_funcs=act_funcs, loss=nn.cross_entropy())
    first=False


# 2クラス分類問題として解く: 出力層はシグモイドの1ユニット
def two_class():
    # 訓練
    net.train(X, T, 
              eta0=0.1,
              max_epoch=100000,
              log_cond=lambda count: count%1000==0,
              how='stdout',
              batch_size=1,
              optimizer='AdaGrad'
    )
    net.test(X, T)

# 決定境界を可視化
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


# 多クラス分類問題として解く: 出力層はソフトマックスの2ユニット
def mul_class():
    T = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # 教師信号は1ofK符号化

    # 各層の活性化関数
    act_funcs = [
        None,
        'sigmoid',
        'softmax'
    ]
    # 各層のニューロン数
    num = [2, 2, 2]
    # mlpオブジェクトを生成
    net = nn.mlp.from_shape(shape=num, act_funcs=act_funcs)

    # 訓練
    net.train(X, T, 
              eta0=0.2,
              max_epoch=1000000,
              log_cond=lambda count: count%1000==0,
              how='stdout',
              batch_size=2,
              optimizer='Momentum'
    )
    # テスト
    net.test(X, T)
