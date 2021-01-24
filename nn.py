"""Multi Layer Perceptron."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import dataclasses
from typing import Type, Sequence, List, Callable



@dataclasses.dataclass
class layer:
    """
    MLPの各層.
    """
    W: np.ndarray       # 前の層から自分への重み行列(結合行列)
    b: np.ndarray       # 前の層から自分へのバイアスベクトル
    h: "act_func"       # 活性化関数
    size: int           # この層に含まれるニューロンの数
    # この層の各ニューロンの活性(内部ポテンシャル)を並べたベクトル
    u: np.ndarray     = dataclasses.field(init=False, default=0.0)
    # 各ニューロンの出力z = h(u)を並べたベクトル
    z: np.ndarray     = dataclasses.field(init=False, default=0.0)
    # 各ニューロンの誤差を並べたベクトル
    delta: np.ndarray = dataclasses.field(init=False, default=0.0)
    # パラメータに関する損失関数の勾配 dJdW, dJ/db
    dJdW: np.ndarray  = dataclasses.field(init=False, default=0.0)
    dJdb: np.ndarray  = dataclasses.field(init=False, default=0.0)
    ## 連結リスト的な表現
    # 前のlayerへの参照
    prev: "layer"     = dataclasses.field(default=None)
    # 後ろのlayerへの参照
    next: "layer"     = dataclasses.field(default=None)
    
    def is_first(self) -> bool:
        return (self.prev is None) and (self.next is not None)
    
    def is_last(self) -> bool:
        return (self.prev is not None) and (self.next is None)

    def is_hidden(self) -> bool:
        return (self.prev is not None) and (self.next is not None)

    def is_unconnected(self) -> bool:
        return (self.prev is None) and (self.next is None)

    def is_connected(self) -> bool:
        return not self.is_unconnected()

    def fire(self, input:np.ndarray) -> np.ndarray:
        """
        inputを入力として層内のニューロンを発火させる.
        
        Parameter
        ---------
        input: 層内の各ニューロンへの入力信号を横に並べたベクトル

        Return
        ------
        inputを入力として活性self.uと出力self.zを更新したのち、self.zへの参照を返す
        """
        self.u = input @ self.W + self.b
        self.z = self.h( self.u )
        return self.z
        
    def set_z(self) -> np.ndarray:
        """
        入力層が現在保持している信号zを使って、ネットワークの入力層から自分自身まで信号を順伝播させる.
        """
        if self.is_first():
            return self.z
        else:
            return self.fire(self.prev.set_z())

    def set_delta(self) -> np.ndarray:
        """
        出力層が現在保持している誤差deltaを使って、ネットワークの出力層から自分自身まで誤差を逆伝播させる.
        """
        if self.is_last():
            return self.delta
        else:
            self.delta = self.h.val2deriv(self.z) * (self.next.set_delta() @ self.next.W.T)
            return self.delta


class conv_layer(layer):
    """畳み込み層"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class pool_layer(layer):
    """プーリング(サブサンプリング)層"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


    
@dataclasses.dataclass
class mlp:
    """
    多層パーセプトロン(MLP: Multi Layer Perceptron). 
    """
    layers: Sequence[layer] # 各層を表すlayerオブジェクトを並べた配列
    loss: "loss_func" = dataclasses.field(default=None) # 損失関数

    def __post_init__(self):
        ## ネットワークの損失関数を設定
        if self.loss is None:
            self.loss = self[-1].h.loss()
        self.loss.net = self
        self.shape = tuple(l.size for l in self)

        ## 層間の連結リストとしての構造を構築する
        for l in range(1, len(self)):
            # 0層目と1層目, 1層目と2層目, ..., L-1層目とL層目とを連結する
            self[l].prev = self[l-1]
            self[l-1].next = self[l]

        ## 入力層の不要なメンバ変数はNoneにする.
        self[0].W = self[0].b = self[0].u = self[0].delta = self[0].h = self[0].dJdW = self[0].dJdb = None

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, x) -> np.ndarray:
        self.forward_prop(x)
        return self[-1].z.copy()

    def copy(self) -> 'mlp':
        layers = []
        for l in self:
            if l.is_first():
                layers.append(layer(size=l.size, W=None, b=None, h=type(l.h)()))
            else:
                layers.append(layer(size=l.size, W=l.W.copy(), b=l.b.copy(), h=type(l.h)()))
        return type(self)(layers, loss=type(self.loss)())
    
    @classmethod
    def from_shape(cls, shape: Sequence[int], act_funcs: Sequence["act_func"], loss=None, sigmas=None) -> 'mlp':
        """
        各層のニューロン数を表す配列と活性化関数を表す配列からlayerオブジェクトとmplオブジェクトを生成する。
        Parameters
        ----------
        shape: list
             ネットワークの各層内にいくつのニューロンを生成するか
        """
        n_layer = len(shape)

        # make sure sigmas is a list
        if not hasattr(sigmas, '__iter__'):
            sigmas = [sigmas for _ in range(n_layer)]
        sigmas = list(sigmas)
        
        layers = [layer(size=shape[0], W=None, b=None, h=None)]
        
        for l in range(1, n_layer):
            if sigmas[l]:
                sigma = sigmas[l]
            elif isinstance(ACTIVATIONS[act_funcs[l]], ReLU):
                sigma = np.sqrt(2.0 / shape[l-1]) # Heの初期値
            else:
                sigma = np.sqrt(1.0 / shape[l-1]) # Xavierの初期値

            W_shape = (shape[l-1], shape[l])
            b_shape = (1, shape[l])

            layers.append(
                layer(
                    size=shape[l],
                    W=np.random.normal(scale=sigma, size=W_shape),
                    b=np.zeros(shape=b_shape),
                    h=ACTIVATIONS[act_funcs[l]]()
                )
            )
        return cls(layers, loss=loss)
    
    def forward_prop(self, x:np.ndarray) -> None:
        """順伝播. xは入力ベクトル. ミニバッチでもOK"""
        self[0].z = x
        return self[-1].set_z()
            
    def back_prop(self, t:np.ndarray) -> None:
        """教師信号ベクトルtをもとに誤差逆伝播法による学習を行う."""
        self[-1].delta = self[-1].z - t # 出力層の誤差はすぐに計算できる.
        self[1].set_delta()             # それを使って誤差を順次前の層へ逆伝播させていく.

    def set_gradient(self) -> None:
        """順伝播と逆伝播により出力zと誤差deltaが計算済みであることを前提に、
        パラメータに関する損失関数の勾配を計算する. 

        Parameters
        ----------
        dJdW, dJdb : array-like of length len(self)
            The locations into which the result is stored.
        """
        n_sample = self[0].z.shape[0]
        for l in range(1, len(self)):
            self[l].dJdW = (self[l-1].z.T @ self[l].delta) / n_sample
            self[l].dJdb = np.mean(self[l].delta, axis=0, keepdims=True)

            
    def train(self,
              X:np.ndarray,
              T:np.ndarray,
              eta:float=0.005,
              optimizer='AdaGrad',
              max_epoch:int=10000,
              batch_size=1,
              log_cond:Callable=lambda count: True,
              color='tab:blue',
              how_to_show='both') -> 'logger':
        """
        訓練データ集合(X: 入力, T: 出力)をネットワークに学習させる. エポック数がmax_iterを超えるか、
        シグナルSIGINT(Ctrl+C)が送出されると学習を打ち切る。

        Parameters
        ----------
        X : array of shape (n_sample, n_feature)
            訓練データセットの入力群.
        T : array of shape (n_sample, n_target)
            訓練データセットの出力群.
        eta: float
            学習係数(の初期値)を表すパラメータ.
        optimizer: {'SGD', 'AdaGrad', 'Momentum'(, 'Adam', 'LBFGS')}
            学習に用いる最適化手法.
        max_epoch:
            最大反復エポック数.
        batch_size: ミニバッチ学習に用いるミニバッチのサイズ. 
        log_cond: カウンタ変数m, iがどんな条件を満たしたときに損失関数の値などをlogに記録するか

        Returns
        -------
        log: 学習の途中経過を記録したloggerオブジェクト
        """
        # 途中経過の記録
        log = logger(
            net=self,
            cond=log_cond,
            n_sample=len(X),
            batch_size=batch_size,
            how_to_show=how_to_show,
            color=color
        )
        # パラメータ更新器
        optimizer = OPTIMIZERS[optimizer](net=self, eta0=eta)
        
        try:            
            for epoch in range(max_epoch):
                for X_mini, T_mini in minibatch_iter(X, T, batch_size):
                    self.forward_prop(X_mini)      # 順伝播
                    self.back_prop(T_mini)         # 逆伝播
                    self.set_gradient()            # パラメータに関する損失の勾配を求める
                    optimizer.update()             # 勾配法による重み更新
                    log()                          # ログ出力
    
        except KeyboardInterrupt:
            pass

        log.end()
        return log
    
    def test(self, X:np.ndarray, T:np.ndarray, log:'logger'=None) -> None:
        """
        テストデータ集合(X: 入力, T: 出力)を用いて性能を試験し、正解率を返す.
        selfは分類器と仮定する. 
        """
        correct = 0
        n_sample = len(X)
        if self[-1].size > 1:
            for i in range(n_sample):
                if np.argmax(T[i]) == np.argmax(self(X[[i]])):
                    correct += 1
        else:
            for i in range(n_sample):
                ans = 1 if self(X[[i]]) > 0.5 else 0
                if T[i] == ans:
                    correct += 1

        rate = correct / n_sample * 100
        print(f"{rate:.2f} % correct")
        if isinstance(log, logger):
            log.rate = rate


            
class minibatch_iter:
    def __init__(self, X : np.ndarray, T : np.ndarray, batch_size : int):
        if len(X) != len(T):
            raise ValueError("'X' and 'T' must have the same length.")
        self.n_sample = len(X)
        self.batch_size = batch_size
        shuffle_idx = np.random.permutation(self.n_sample)
        self.X = X[shuffle_idx]
        self.T = T[shuffle_idx]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.X) <= 0:
            raise StopIteration
        X_mini = self.X[:self.batch_size]
        T_mini = self.T[:self.batch_size]
        self.X = self.X[self.batch_size:]
        self.T = self.T[self.batch_size:]
        return X_mini, T_mini
        
            
        
class _optimizer_base:
    """Optimization solvers for MLP training. 
    
    Attributes
    ----------
    net : mlp
        The MLP object whose paramters should be optimized.
    eta0 : float
        The initial value of learning rate.
    eta : float
        The current value of learning rate.
    """
    def __init__(self, net, eta0):
        self.net = net
        self.eta0 = eta0
        self.eta = self.eta0
        # パラメータの1ステップあたり更新量
        self.dWs = [None] + [0 for _ in range(1, len(self.net))] # 重み行列用
        self.dbs = [None] + [0 for _ in range(1, len(self.net))] # バイアス用

    def get_update(self):
        """Set new values to`dWs` and `dbs`. Override this method when you create a new subclasses.
        """
        raise NotImplementedError

    def update(self):
        self.get_update()
        for layer, dW, db in zip(self.net[1:], self.dWs[1:], self.dbs[1:]):
            layer.W += dW
            layer.b += db

        
class SGD(_optimizer_base):
    def get_update(self):
        for l in range(1, len(self.net)):
            self.dWs[l] = -self.eta * self.net[l].dJdW
            self.dbs[l] = -self.eta * self.net[l].dJdb

            
class AdaGrad(_optimizer_base):
    def __init__(self, net, eta0, eps=1e-7):
        super().__init__(net, eta0)
        # dJ/dWやdJ/dbの二乗和を保持する変数
        self.h_W = [None] + [0 for _ in range(1, len(self.net))] # 重み行列用
        self.h_b = [None] + [0 for _ in range(1, len(self.net))] # バイアス用
        # self.etaの更新時のゼロ除算を防止するための微小な数
        self.eps = eps
        # 実装の都合上、学習係数を重み用とバイアス用に分ける
        self.eta_W = [None] + [self.eta0 for _ in range(1, len(self.net))]
        self.eta_b = [None] + [self.eta0 for _ in range(1, len(self.net))]

    def get_update(self):        
        for l in range(1, len(self.net)):
            self.h_W[l] += self.net[l].dJdW * self.net[l].dJdW
            self.h_b[l] += self.net[l].dJdb * self.net[l].dJdb

            self.eta_W[l] = self.eta0 / (np.sqrt(self.h_W[l]) + self.eps)
            self.eta_b[l] = self.eta0 / (np.sqrt(self.h_b[l]) + self.eps)

            self.dWs[l] = -self.eta_W[l] * self.net[l].dJdW
            self.dbs[l] = -self.eta_b[l] * self.net[l].dJdb


class Momentum(_optimizer_base):
    def __init__(self, net, eta0, momentum=0.9):
        super().__init__(net, eta0)
        self.momentum = momentum
        
    def get_update(self):
        for l in range(1, len(self.net)):
            self.dWs[l] = -self.eta * self.net[l].dJdW + self.momentum * self.dWs[l]
            self.dbs[l] = -self.eta * self.net[l].dJdb + self.momentum * self.dbs[l]


class Adam(_optimizer_base):
    def __init__(self):
        raise NotImplementedError


class LBFGS(_optimizer_base):
    """Limited-Memory BFGS."""
    def __init__(self):
        raise NotImplementedError
        

OPTIMIZERS = {
    'SGD'      : SGD,
    'AdaGrad'  : AdaGrad,
    'Momentum' : Momentum,
    'Adam'     : Adam,
    'LBFGS'    : LBFGS
}



@dataclasses.dataclass
class act_func:
    """各層の活性化関数"""
    param: float = 1.0
    loss: Type["loss_func"] = dataclasses.field(default=None) # 出力層の活性化関数として用いた場合の対応する損失関数クラス

    def __call__(self, u):
        """活性化関数の値h(u)そのもの."""
        raise NotImplementedError

    def val2deriv(self, z):
        """活性化関数の関数値z = h(u)の関数として導関数h\'(u)の値を計算する."""
        raise NotImplementedError

    
class sigmoid(act_func):
    """シグモイド関数"""
    def __init__(self):
        super().__init__()
        self.loss = cross_entropy
        
    def __call__(self, u):
        return 0.5 * (1.0 + np.tanh(0.5 * self.param * u))
    
    def val2deriv(self, z, th=0.001):
        return z * (1.0 - z) # np.maximum(z, th) * (1.0 - np.minimum(z, 1-th))

    
class linear(act_func):
    """線形関数"""
    def __init__(self):
        super().__init__()
        self.loss = mean_square

    def __call__(self, u):
        return u
    
    def val2deriv(self, z):
        return 1.0

    
class ReLU(act_func):
    """ReLU"""
    def __call__(self, u):
        return np.maximum(u, 0.0)
    
    def val2deriv(self, z):
        return np.where(z > 0.0, 1.0, 0.0)

    
class softmax(act_func):
    """ソフトマックス関数"""
    def __init__(self):
        super().__init__()
        self.loss = mul_cross_entropy
        
    def __call__(self, u):
        tmp = u - u.max()
        return np.exp(tmp) / np.sum(np.exp(tmp), axis=1, keepdims=True)
    
    def val2deriv(self, z):
        return Exception("出力層の活性化関数の微分は使わないはず")


ACTIVATIONS = {
    'identity' : linear,
    'linear'   : linear,
    'sigmoid'  : sigmoid,
    'logistic' : sigmoid,
    'relu'     : ReLU,
    'ReLU'     : ReLU,
    'softmax'  : softmax
}


    
@dataclasses.dataclass
class loss_func:
    net: mlp = dataclasses.field(default=None, repr=False)
    
    """損失関数"""    
    def __call__(self, x, t):
        """
        損失関数の値.

        Parameter(s)
        ------------
        x : 教師データの入力
        t : 教師データの出力
        """
        raise NotImplementedError

    def error(self, t):
        """
        出力層の誤差delta = 出力層の内部ポテンシャルuによる微分
        """
        return self.net[-1].z - t


class mean_square(loss_func):
    def __call__(self, x, t):
        if x is not None:
            self.net.forward_prop(x)
        batch_size = len(t)
        SSE = 0.5 * np.linalg.norm(self.net[-1].z - t)**2
        return SSE / batch_size
    
class cross_entropy(loss_func):
    def __call__(self, x, t):
        if x is not None:
            self.net.forward_prop(x)

        batch_size = len(t)
        z = self.net[-1].z
        return -(t * np.log(z) + (1.0 - t) * np.log(1 - z)).sum() / batch_size

    
class mul_cross_entropy(cross_entropy):
    def __call__(self, x, t):
        if x is not None:
            self.net.forward_prop(x)
        batch_size = len(t)
        return -np.sum((t * np.log(self.net[-1].z + 1e-7))) / batch_size


    
@dataclasses.dataclass
class logger:
    """学習経過の記録"""
    net : mlp             = dataclasses.field(default=None)
    loss: Sequence[float] = dataclasses.field(default_factory=list)
    count: Sequence[int]  = dataclasses.field(default_factory=list)
    iterations : int      = dataclasses.field(init=False, default=0)
    rate: float           = dataclasses.field(default=None)
    time: float           = dataclasses.field(default=None)
    cond: Callable        = dataclasses.field(default=lambda count: True)
    n_sample : int        = dataclasses.field(default=None)
    batch_size: int       = dataclasses.field(default=1)
    how_to_show: str      = dataclasses.field(default='plot')
    color : str           = dataclasses.field(default='tab:blue')
    base : int            = dataclasses.field(default=20)

    def __post_init__(self):
        # 学習開始時間を記録
        self.t0 = time.time()
        # 1エポックあたりiteration数
        self.iter_per_epoch = int(np.ceil(self.n_sample / self.batch_size))
        # 損失の変化をどう表示するか
        if self.how_to_show == 'both':
            self.plot, self.stdout = True, True
        elif self.how_to_show == 'plot':
            self.plot, self.stdout = True, False
        elif self.how_to_show == 'stdout':
            self.plot, self.stdout = False, True
        elif self.how_to_show == 'off':
            self.plot, self.stdout = False, False
        else:
            raise ValueError("logger.how_to_show must be either of the followings: 'both', 'plot', 'stdout' or 'off'")

        if self.plot:
            # 損失のグラフをリアルタイムで
            self.fig, self.ax = plt.subplots(constrained_layout=True)
            self.ax.set(xlabel="iterations", ylabel="loss")
            self.secax = self.ax.secondary_xaxis(
                'top',
                functions=(lambda count: count/self.iter_per_epoch,
                           lambda epoch: epoch*self.iter_per_epoch)
            )
            
            self.secax.xaxis.set_major_locator(ticker.AutoLocator())
            self.secax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            self.secax.set_xlabel('epoch')

            self.secax.tick_params(axis='x', which='major', length=10)
            self.secax.set_xlim(left=0)
            self.ax.grid(axis='y', linestyle='--')
            plt.ion()

    def __call__(self) -> None:
        if self.cond(self.iterations):
            # log出力
            T_mini = self.net[-1].z - self.net[-1].delta
            self.loss.append(self.net.loss(None, T_mini))
            self.count.append(self.iterations)
            # 現在のエポック
            epoch = self.iterations // self.iter_per_epoch
            # epoch内で何番目のiterationか
            idx_iter = self.iterations % self.iter_per_epoch
            # epoch内で何番目のパターンか
            idx_sample = min(
                (idx_iter + 1) * self.batch_size,
                self.n_sample
            )
            
            logstr = f"Epoch {epoch:3}, Pattern {idx_sample:5}/{self.n_sample}: Loss = {self.loss[-1]:.3e}"
            
            if self.stdout:
                print(logstr)

            if self.plot:
                self.ax.set_xlim(0, (int(np.ceil((epoch+1e-4)/self.base))*self.base) * self.iter_per_epoch)
                self.ax.plot(self.count[-2:], self.loss[-2:], c=self.color)
                self.ax.set_title(logstr)
                plt.show()
                plt.pause(0.1)

        self.iterations += 1

    def end(self) -> None:
        self.tf = time.time()
        self.time = self.tf - self.t0
    
    def show(self, ax=None, semilog=False, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.count, self.loss)
        ax.set(xlabel="iteration", ylabel="loss", *args, **kwargs)
        if semilog:
            ax.set_yscale("log")

    def to_file(self, fname) -> None:
        with open(fname, "w") as f:
            f.write(repr(self))
            
    @classmethod
    def from_file(cls, fname):
        with open(fname) as f:
            return eval("cls" + f.read().replace("logger", ""))


def numerical_gradient(func, x, dx=1e-7):
    """Numerically compute the gradient of given function with central difference.

    This code was originally written by Koki SAITO
    (https://github.com/oreilly-japan/deep-learning-from-scratch/blob/89eadd0804af4de07e13fc6a478d591124f89312/common/gradient.py#L34-L52)
    and slightly editted by Ryota USHIO.

    Parameters
    ----------
    func : Callable
        The function to be differentiated.
    x : float or np.ndarray
        The point that ``func`` is differentiated at.
    dx : float
    """
    f = func
    h = dx
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def _numerical_loss_grad(net, layer, X, T, dW=1e-7, db=1e-7):
    """Numerically computes gradient of the given MLP's loss function 
    with respect to parameters of the specified layer.

    Note that outputs which correspond current parameters have already computed 
    by forward propagation.

    Parameters
    ----------
    net : mlp
    layer : int
    X : array of shape (n_sample, n_feature)
    T : array of shape (n_sample, n_target)
    """
    return (numerical_gradient(lambda W: net.loss(X, T), net[layer].W, dW),
            numerical_gradient(lambda b: net.loss(X, T), net[layer].b, db))

def numerical_loss_grad(net, X, T, dW=1e-9, db=1e-9):
    """Numerically computes gradient of the given MLP's loss function 
    with respect to parameters of all the layers.
    """
    dJdW = [None for _ in range(len(net))]
    dJdb = [None for _ in range(len(net))]
    for layer in range(1, len(net)):
        dJdW_l, dJdb_l = _numerical_loss_grad(net, layer, X, T, dW, db)
        dJdW[layer] = dJdW_l
        dJdb[layer] = dJdb_l
    return dJdW, dJdb

    
def backprop_loss_grad(net, X, T):
    """Computes gradient of the given MLP's loss function with respect to parameters
    of the specified layer BY BACK-PROP.
    """
    net.forward_prop(X)
    net.back_prop(T)
    net.set_gradient()
    return [l.dJdW for l in net], [l.dJdb for l in net]

    
def gradient_check(net, X, T):
    """A gradient checker for MLP.

    Certify that the implementation of back-propation is correct by comparing 
    the results of gradient computation obtained by two different methods
    (numerical differentiation and back-prop), 
    """
    dJdW_bp, dJdb_bp = backprop_loss_grad(net, X, T)
    dJdW_nd, dJdb_nd = numerical_loss_grad(net, X, T)

    for l in range(1, len(net)):
        print(dJdW_bp[l].shape, dJdW_nd[l].shape)
        print(dJdb_bp[l].shape, dJdb_nd[l].shape)
        print(f"layer{l} weight : {np.average(np.abs(dJdW_bp[l] - dJdW_nd[l]))}, {dJdW_bp[l] / dJdW_nd[l]}")
        print(f"layer{l} bias   : {np.average(np.abs(dJdb_bp[l].reshape(1,-1) - dJdb_nd[l]))}, {dJdb_bp[l] / dJdb_nd[l]}")
