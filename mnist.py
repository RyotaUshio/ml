import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import keras.datasets
from typing import Sequence, List, Callable
from mpl_toolkits.axes_grid1 import Size, Divider



def image_classifier(X_train, T_train,
                     hidden_shape=[10],
                     hidden_act='ReLU',
                     out_act='softmax',
                     loss=None,
                     eta0:float=0.1,
                     max_epoch:int=300,
                     batch_size=200,
                     optimizer='AdaGrad',
                     dropout=False,
                     *args, **kwargs):
    """
    画像データセット用のMLPのインターフェース. 
    10クラス分類用のmlpオブジェクトを生成し、MNISTデータセットを学習させ、そのmlpオブジェクトとlogを返す。
    Parameters
    ----------
    hidden_shape: Sequence of int
    hidden_act : {'identity' = 'linear, 'sigmoid' = 'logistic', 'relu' = 'ReLU', 'softmax'}
    others:
         mlpオブジェクトのtrainメソッドの引数

    Returns
    -------
    net: nn.mlp
         学習済みmlpオブジェクト
    log: nn.logger
         学習の途中経過などを記録したnn.loggerオブジェクト
    """
    net = make_clf(shape=[X_train[0].size] + hidden_shape + [T_train[0].size],
                   hidden_act=hidden_act,
                   out_act=out_act,
                   loss=loss,
                   dropout=dropout)
    # 学習を実行
    net.train(X_train, T_train, 
              eta0=eta0,
              max_epoch=max_epoch,
              batch_size=batch_size,
              optimizer='AdaGrad',
              *args, **kwargs
    )
    # 学習済みのmlpオブジェクトを返す
    return net


def make_clf(shape, hidden_act='sigmoid', out_act='softmax', dropout=False, *args, **kwargs):
    """hidden_act : {'identity' = 'linear, 'sigmoid' = 'logistic', 'relu' = 'ReLU', 'softmax'}
    """
    # 各層の活性化関数
    act_funcs = (
        [None] +
        [hidden_act for _ in range(len(shape[1:-1]))] +
        [out_act]
    )
    # mlpオブジェクトを生成
    net = nn.mlp.from_shape(shape=shape, act_funcs=act_funcs, *args, **kwargs)
    if dropout:
        net = nn.dropout_mlp.from_mlp(net, dropout)
    return net

    
def hidden_test(net, j):
    w = normalize(net[1].W[:, j])
    y = np.argmax(net(w))
    plt.imshow(w.reshape(28,28), cmap=plt.cm.binary)
    plt.title(f'{y}?')
    print(net(w))

def last_test(net, j):
    w = normalize(net[-1].W[:, j])
    # y = np.argmax(net(w))
    plt.imshow(w[2:-2].reshape(4,4), cmap=plt.cm.binary)
    # plt.title(f'{y}?')
    # print(net(w))
    
def hidden_show(net, figsize=(8-1/4, 11-3/4)):
    size = figsize
    rect = (0.15, 0.15, 0.79, 0.79)
    pad_width = size[0]*.05/9.0 * rect[2]
    pad_height = pad_width*5
    im_height = im_width = (size[0] * rect[2] - pad_width*9)/10
    bar_height = (size[1]*rect[2] - pad_height*10 - im_height)/10
    bar_width = size[0] * rect[2]
    
    fig = plt.figure(figsize=size)

    # fixed size in inch
    horiz = ([Size.Fixed(im_width), Size.Fixed(pad_width)] * 10)[:-1]
    vert =  [Size.Fixed(im_height)] + [Size.Fixed(pad_height), Size.Fixed(bar_height)]*10
    vert[1] = Size.Fixed(pad_height*2)

    # divide the axes rectangle into grid whose size is specified by horiz * vert
    divider = Divider(fig, rect, horiz, vert, aspect=False)
    plt.ion()

    reps = rep()
    w_min, w_max = np.inf, -np.inf
    for last in range(10):
        # last: 出力層の各ニューロンの番号
        ax = fig.add_axes(rect, label=f"bar{last}")
        ax.set_axes_locator(divider.new_locator(nx=0, nx1=-1, ny=20-2*last))
        w = net[-1].W[:, last]
        if w.max() > w_max:
            w_max = w.max()
        if w.min() < w_min:
            w_min = w.min()
        ax.bar(x=range(10), height=w)
        ax.plot(np.linspace(-0.5, 9.5, 10), np.full(10, 0), linestyle='--', color='k', linewidth=.8)
        ax.xaxis.set_visible(False)

    w_range = max(abs(w_max), abs(w_min))
    for last, ax in zip(range(10), fig.axes):
        ax.set(xlim=(-0.5, 9.5), ylim=(-w_range, w_range), ylabel=f'{last}')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_visible(True)

    for hidden in range(10):
        # hidden: 中間層の各ニューロンの番号
        ax = fig.add_axes(rect, label=f"image{hidden}")
        ax.set_axes_locator(divider.new_locator(nx=hidden*2, ny=0))
        w = normalize(net[1].W[:, hidden])
        imshow(w, ax=ax)

    fontsize = 'large'
    fig.text(rect[0]+0.5*rect[2], 0.12, 'Hidden Units', ha='center', va='center', fontsize=fontsize)
    fig.text(0.06, rect[1]+0.5*rect[3], 'Output Units', ha='center', va='center', rotation='vertical', fontsize=fontsize)
    
def rep():
    if cur_data != keras.datasets.mnist:
        raise Exception(f"expected cur_data == keras.datasets.mnist, but now cur_data == {cur_data}")
    reps = []
    for i, j in zip(range(10), [0, 0, 2, 0, 16, 19, 5, 0, 2, 2]):
        reps.append(X_train[vec2num(T_train) == i][j])
    return np.array(reps)
