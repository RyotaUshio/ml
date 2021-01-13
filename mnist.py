import numpy as np
import matplotlib.pyplot as plt
import keras.datasets.mnist
from typing import Sequence, List, Callable
from importlib import reload
import nn
reload(nn)

def one_of_K(labels:np.ndarray):
    """
    正解ラベルの集合labelsを1 of K符号化法によりベクトル化する
    """
    I = np.identity(labels.max() - labels.min() + 1)
    return [I[i] for i in labels]

def vec2num(one_of_K:Sequence):
    """1 of K符号化されたベクトルone_of_Kをクラス番号に変換する"""
    return np.argmax(one_of_K)

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
    noise_num = int(image.size * prob)
    idx = np.random.randint(0, image.size, noise_num)
    image[idx] = np.random.rand(noise_num)

def add_noise_all(images, prob):
    cp = images.copy()
    for image in cp:
        add_noise(image, prob)
    return cp

def img_show(image, ax, shape=None):
    if shape is None:
        tmp = int(np.sqrt(image.size))
        shape = (tmp, tmp)
    ax.imshow(image.reshape(shape), cmap=plt.cm.binary)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def down_sample(image, rate):
    hi = int(np.ceil(image.shape[0]/rate))
    wid = int(np.ceil(image.shape[1]/rate))
    ret = np.zeros((hi, wid))
    for i in range(hi):
        for j in range(wid):
            ret[i][j] = image[i*rate][j*rate]
    return ret
    
def mnist(prob:float=0, n_layer:int=1, n_neuron:int=196,
          eta:float=0.005,
          eps:float=0.02,
          max_iter:int=20,
          log_cond:Callable=lambda m, i: i%1000==0,
          *args, **kwargs):
    """
    10クラス分類用のmlpオブジェクトを生成し、MNISTデータセットを学習させ、そのmlpオブジェクトとlogを返す。
    Parameters
    ----------
    prob:float
         ノイズ割合(0.0--1.0)
    n_layer:int
         中間層の層数
    n_neuron:int
         中間層1層あたりのニューロン数
    others:
         mlpオブジェクトのtrainメソッドの引数

    Returns
    -------
    net: nn.mlp
         学習済みmlpオブジェクト
    log: nn.logger
         学習の途中経過などを記録したnn.loggerオブジェクト
    """
    # ノイズを付加した入力画像データを作成
    X_train_, X_test_ = add_noise_all(X_train, prob), add_noise_all(X_test, prob)
    # 各層の活性化関数
    act_funcs = [nn.linear()] + [nn.sigmoid() for _ in range(n_layer)] + [nn.softmax()]
    # 入力次元数
    d = X_train_[0].size
    # 出力次元数
    K = 10
    # 各層のニューロンの数
    num = [d] + [n_neuron for _ in range(n_layer)] + [K]
    # mlpオブジェクトを生成
    net = nn.mlp.from_num(num=num, act_funcs=act_funcs, loss=nn.mul_cross_entropy())

    # 学習を実行
    log = net.train(X_train_, T_train, 
                    eta=eta,
                    eps=eps,
                    max_iter=max_iter,
                    log_cond=log_cond, 
                    *args, **kwargs
    )

    # 性能試験
    print("train")
    net.test(X_train_, T_train)
    print("test")
    net.test(X_test_, T_test, log=log)

    # mlpとログを返す
    return net, log


first=True
if first:
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    X_train, T_train = convert(train_images, train_labels, 2)
    X_test, T_test = convert(test_images, test_labels, 2)
    first = False
