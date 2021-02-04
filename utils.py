import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import keras.datasets as datasets
from typing import Sequence, List, Callable
import pickle


DATASETS = {
    'mnist'         : datasets.mnist,
    'fashion_mnist' : datasets.fashion_mnist,
    'cifar10'       : datasets.cifar10,
    'cifar100'      : datasets.cifar100
}

fashion_mnist_labels = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 :	'Pullover',
    3 :	'Dress',
    4 :	'Coat',
    5 :	'Sandal',
    6 :	'Shirt',
    7 :	'Sneaker',
    8 :	'Bag',
    9 :	'Ankle boot'
}

### データを読み込む
def load_data(
        data='mnist',
        negative=False,
        n_target=10
        ):
    (train_images, train_labels), (test_images, test_labels) = DATASETS[data].load_data()
    X_train, T_train = convert(train_images, train_labels, n_target)
    X_test, T_test = convert(test_images, test_labels, n_target)

    if negative:
        X_train = 1.0 - X_train
        X_test = 1.0 - X_test

    return (X_train, T_train), (X_test, T_test)


## データセットの整形用の諸関数

def one_of_K(labels:Sequence[int], n_target=None):
    """
    正解ラベルの集合labelsを1 of K符号化法によりベクトル化する
    """
    if n_target is None:
        n_target = len(np.unique(labels))
    I = np.identity(n_target)
    return I[labels]

def vec2label(one_of_K:Sequence):
    """1 of K符号化されたベクトルone_of_Kをクラス番号に変換する"""
    ndim = one_of_K.ndim
    if ndim <= 1:
        return np.argmax(one_of_K)
    elif ndim == 2:
        return np.argmax(one_of_K, axis=1)
    else:
        raise ValueError("Expected an array-like of ndim <= 2, got {ndim}")

vec2digit = vec2label # alias

def normalize(x:np.ndarray, lower=None, upper=None):
    """
    入力パターンxの各要素を0-1に正規化する.
    Parameters
    ----------
    x: 入力パターン(の集合)
    range: (最小値, 最大値)の配列
    """
    if lower is None:
        lower = x.min()
    if upper is None:
        upper = x.max()
    return (x - lower) / (upper - lower)

def convert(inputs, labels, n_target, flatten=True):
    """
    データセットをニューラルネットワークに入力できる形に変換する.
    """
    if flatten:
        X = np.array([input.flatten() for input in inputs])
    else:
        X = np.array(inputs)
    X = normalize(X)
    T = one_of_K(labels, n_target)
    return X, T

def imshow(image, ax=None, shape=None):
    """画像imageをaxに表示・可視化する. """
    if ax is None:
        fig, ax = plt.subplots()
    if image.ndim == 1:
        image = vec2img(image, shape=shape)
    ax.imshow(image, cmap=plt.cm.gray)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def vec2img(vec, shape=None):
    """1次元の特徴ベクトルを2次元配列に戻す"""
    if shape is None:
        tmp = int(np.sqrt(vec.size))
        shape = (tmp, tmp)
    return vec.reshape(shape)


## 分類器の出力を整形する

def prob2label(x):
    labels = np.argmax(x, axis=1)
    return labels

def prob2one_of_K(x):
    maximum = np.max(x, axis=1, keepdims=True)
    mask = (x >= maximum)
    return np.ones(x.shape) * mask

def is_one_of_K(T):
    return (np.all((T == 0) | (T == 1)) and np.all(np.sum(T, axis=1) == 1))

def check_twodim(a:np.ndarray=None):
    """Make sure an array is two-dimensinal.
    """
    if a is None:
        raise TypeError("You have to pass an array-like, not None")
    a = np.array(a)
    if a.ndim <= 1:
        return a.reshape((1, -1))
    elif a.ndim >= 3:
        raise ValueError("A three dimensional array was passed.")
    return a


def save(obj, filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load(filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def label_rep(T : np.ndarray):
    if is_one_of_K(T):
        return 'one of K'
    elif T.size == T.shape[0]:
        return 'digit'
    else:
        raise ValueError(
            "Label must be a digit or a 1-of-K encoded vector."
        )

    
def estimate_params(
        X : np.ndarray, T : np.ndarray,
        mean : bool=True, cov : bool=True, prior : bool=True
):
    """Maximum likelihodd estimation of mean & covariance of the normal distribution,
    and prior probability of each class.

    Parameters
    ----------
    X, T : np.ndarray of shape (n_sample, n_feature) & (n_sample, n_target)
        The dataset whose parameters are computed.
    mean, cov, prior : bool, defalut=True
        Whether to compute each parameter or not.

    Returns
    -------
    means, covs, priors : np.ndarray
        Arrays that contain computed parameter of each class.
    """
    _label_rep = label_rep(T)
    n_sample = len(T)

    unq, idx = np.unique(T, axis=0, return_inverse=True)
    sortidx = np.argsort(idx)
    idx = idx[sortidx]
    X = X[sortidx]
    T = T[sortidx]
    
    means = []
    covs = []
    priors = []
    
    for i in range(len(unq)):
        ith = i if _label_rep == 'digit' else len(unq)-i-1
        class_i = X[idx == ith]

        if mean:
            mean_i = np.mean(class_i, axis=0)
            means.append(mean_i)
        if cov:
            cov_i = np.cov(class_i, rowvar=False)
            covs.append(cov_i)
        if prior:
            priors.append(len(class_i) / n_sample)
        
    return np.array(means), np.array(covs), np.array(priors)
