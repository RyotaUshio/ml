import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import keras.datasets
from typing import Sequence, List, Callable
from importlib import reload
import nn
reload(nn)
from mpl_toolkits.axes_grid1 import Size, Divider


### データを読み込む
if any([name not in globals() for name in ['cur_data', 'cur_rate']]):
    cur_data = cur_rate = None

def load(
        rate = 1,
        data = keras.datasets.mnist,
        negative=False
        ):
    global cur_rate, cur_data
    
    if any([name not in globals() for name in [
            'train_images',
            'train_labels',
            'test_images',
            'test_labels',
            'X_train',
            'T_train',
            'X_test',
            'T_test']]) or rate != cur_rate or data != cur_data:

        global train_images, train_labels, test_images, test_labels, X_train, T_train, X_test, T_test

        (train_images, train_labels), (test_images, test_labels) = data.load_data()
        X_train, T_train = convert(train_images, train_labels, rate)
        X_test, T_test = convert(test_images, test_labels, rate)

        if negative:
            X_train = 1.0 - X_train
            X_test = 1.0 - X_test

        cur_data = data
        cur_rate = rate


## データセットの整形用の諸関数

def one_of_K(labels:np.ndarray):
    """
    正解ラベルの集合labelsを1 of K符号化法によりベクトル化する
    """
    I = np.identity(labels.max() - labels.min() + 1)
    T = np.array([I[int(i)] for i in labels])
    return T

def vec2num(one_of_K:Sequence):
    """1 of K符号化されたベクトルone_of_Kをクラス番号に変換する"""
    ndim = one_of_K.ndim
    if ndim == 1:
        return np.argmax(one_of_K)
    elif ndim == 2:
        return np.array([vec2num(t) for t in one_of_K])
    else:
        raise ValueError(f"expected one_of_K.ndim <= 2, got {ndim}")

def normalize(x:np.ndarray, range:Sequence=None):
    """
    入力パターンxの各要素を0-1に正規化する.
    Parameters
    ----------
    x: 入力パターン(の集合)
    range: (最小値, 最大値)の配列
    """
    if range is None:
        range = (x.min(), x.max())
    return (x - range[0]) / (range[1]-range[0])

def convert(images, labels, rate=1):
    """
    画像の集合imagesと正解ラベルの集合labelsを、ニューラルネットワークに入力できる形に変換する
    """
    X = np.array([down_sample(image, rate).flatten() for image in images])
    X = normalize(X)
    T = one_of_K(labels)
    return X, T

def add_noise(image, prob):
    """画像imageに, [0,1]上の一様分布からサンプリングしたprob %のノイズを付加する. in-placeな処理."""
    if prob == 0:
        return
    noise_num = int(image.size * prob)
    idx = np.random.randint(0, image.size, noise_num)
    image[idx] = np.random.rand(noise_num)

def add_noise_all(images, prob):
    """imagesに含まれるすべての画像imageに対してadd_noiseと同様のノイズを付加し、新しいオブジェクトとして返す. 非in-inplace."""
    cp = images.copy()
    for image in cp:
        add_noise(image, prob)
    return cp

def imshow(image, ax=None, shape=None):
    """画像imageをaxに表示・可視化する. """
    if ax is None:
        ax = plt.gca()
    if image.ndim == 1:
        image = vec2img(image, shape=shape)
    ax.imshow(image, cmap=plt.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def down_sample(image, rate):
    """画像imageを1/rateにダウンサンプリングして返す. """
    if rate == 1:
        return image
    hi = int(np.ceil(image.shape[0]/rate))
    wid = int(np.ceil(image.shape[1]/rate))
    ret = [[image[i*rate][j*rate] for j in range(wid)] for i in range(hi)]
    return np.array(ret)

def is_rgb(image):
    return image.ndim == 3

def is_grayscale(image):
    return image.ndim == 2

def is_flatten(image):
    return image.ndim == 1
    
def image_classifier(hidden_shape=[10],
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
    net = make_clf(hidden_shape=hidden_shape, hidden_act=hidden_act, out_act=out_act, loss=loss, dropout=dropout)
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


def make_clf(hidden_shape, hidden_act='sigmoid', out_act='softmax', dropout=False, *args, **kwargs):
    """hidden_act : {'identity' = 'linear, 'sigmoid' = 'logistic', 'relu' = 'ReLU', 'softmax'}
    """
    # 各層の活性化関数
    act_funcs = (
        [None] +
        [hidden_act for _ in range(len(hidden_shape))] +
        [out_act]
    )
    # 入力次元数
    d = X_train[0].size
    # 出力次元数
    K = T_train[0].size
    shape = [d] + hidden_shape + [K]
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

# def hipass(img, *args, **kwargs):
#     lowpass = ndimage.gaussian_filter(img, *args, **kwargs)
#     return img - lowpass

# def usm(img, k, *args, **kwargs):
#     """Unsharpe Masking."""
#     return (normalize(img + k * hipass(img, *args, **kwargs)) * 255).astype(np.uint64)

def vec2img(vec, shape=None):
    """1次元の特徴ベクトルを2次元配列に戻す"""
    if shape is None:
        tmp = int(np.sqrt(vec.size))
        shape = (tmp, tmp)
    return vec.reshape(shape)
    
def rep():
    if cur_data != keras.datasets.mnist:
        raise Exception(f"expected cur_data == keras.datasets.mnist, but now cur_data == {cur_data}")
    reps = []
    for i, j in zip(range(10), [0, 0, 2, 0, 16, 19, 5, 0, 2, 2]):
        reps.append(X_train[vec2num(T_train) == i][j])
    return np.array(reps)
