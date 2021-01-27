"""Multilayer Perceptron."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import dataclasses
from typing import Type, Sequence, List, Callable
import warnings



class layer:
    """
    MLPの各層.
    """
    def __init__(self, W:Sequence=None, b:Sequence=None, h:'act_func'=None, size:int=None, first:bool=False):
        # for an non-input layer
        if not first:
            # make sure that all matrices and vectors are a two-dimensional numpy.ndarray
            self.W = check_twodim(W)
            self.b = check_twodim(b)
            # validation of the shape of weight matrix and bias vector
            if self.W.shape[1] != self.b.shape[1]:
                raise ValueError(
                    "Expected self.W.shape[1] == self.b.shape[1], got self.W of shape {self.W.shape} and self.b of shape {self.b.shape}"
                )
            # validate and set self.size (number of the neurons included in this layer)
            if (size is not None) and (self.W.shape[1] != size):
                raise ValueError("Incompatible size was specified.")
            self.size = self.W.shape[1]
            # set the activation function
            self.h = act_func.make(h)

        # for an input layer
        else:
            if size is None:
                raise ValueError(
                    "The size of an input layer must be specified because it cannot be inferred from W.shape or b.shape"
                )
            self.size = size
            if not (W == b == h == None):
                warnings.warn(
                    "Your specification of the input layer parameters will be ignored"
                    " because they must be set None."
                )
            self.W = self.b = self.h = None
        
        self.u = self.delta = self.dJdW = self.dJdb = None
        self.prev = self.next = None

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.size} neurons with {self.h.__class__.__name__} activation>"
    
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
        
    def prop_z(self) -> np.ndarray:
        """入力層が現在保持している信号zを使って、ネットワークの入力層から
        自分自身まで信号を順伝播させる.
        """
        if self.is_first():
            return self.z
        else:
            return self.fire(self.prev.prop_z())

    def calc_delta(self, next_delta) -> np.ndarray:
        """次の層における誤差から今の層の誤差を求める.
        """
        self.delta = self.h.val2deriv(self.z) * (next_delta @ self.next.W.T)
        return self.delta

    def prop_delta(self) -> np.ndarray:
        """出力層が現在保持している誤差deltaを使って、ネットワークの出力層から
        自分自身まで誤差を逆伝播させる.
        """
        if self.is_last():
            return self.delta
        else:
            return self.calc_delta(self.next.prop_delta())

    def set_gradient(self) -> None:
        """順伝播と逆伝播により出力zと誤差deltaが計算済みであることを前提に、
        パラメータに関する損失関数の勾配を計算する. 
        """
        batch_size = self.z.shape[0]
        self.dJdW = (self.prev.z.T @ self.delta) / batch_size
        self.dJdb = np.mean(self.delta, axis=0, keepdims=True)

        

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
    多層パーセプトロン(MLP: Multilayer Perceptron). 
    """
    layers: Sequence[layer]                             # 各層を表すlayerオブジェクトを並べた配列
    loss: 'loss_func' = dataclasses.field(default=None) # 損失関数
    log: 'logger' = dataclasses.field(init=False, default=None, repr=False)

    @classmethod
    def from_shape(cls, shape: Sequence[int], act_funcs: Sequence[str], loss=None, sigmas=None) -> 'mlp':
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
        
        layers = [layer(size=shape[0], first=True)]

        for l in range(1, n_layer):
            if sigmas[l]:
                sigma = sigmas[l]
            elif isinstance(act_func.make(act_funcs[l]), (ReLU, LeakyReLU)):
                sigma = np.sqrt(2.0 / shape[l-1]) # Heの初期値
            else:
                sigma = np.sqrt(1.0 / shape[l-1]) # Xavierの初期値

            W_shape = (shape[l-1], shape[l])
            b_shape = (1, shape[l])

            layers.append(
                layer(
                    W=np.random.normal(scale=sigma, size=W_shape),
                    b=np.zeros(shape=b_shape),
                    h=act_funcs[l]
                )
            )

        return cls(layers, loss=LOSSES[loss]())

    @classmethod
    def from_params(cls, weights:Sequence, biases:Sequence, act_funcs:Sequence[str], loss=None, include_first=False):
        """Make a mlp object by specifying paramters(=weight matrices and bias vectors) and 
        activation functions of each layer."""
        if not (len(weights) == len(biases) == len(act_funcs)):
            raise ValueError("'weights', 'biases', and 'act_funcs' must have the same length.")

        if include_first:
            first_layer = layer(
                size=np.asarray(weights[1]).shape[0],
                W=weights[0], b=biases[0], h=act_funcs[0],
                first=True
            )
            weights = weights[1:]
            biases = biases[1:]
            act_funcs = act_funcs[1:]
        else:
            first_layer = layer(size=np.asarray(weights[0]).shape[0], first=True)

        layers = (
            [first_layer]
            +
            [layer(
                W=weight,
                b=bias,
                h=act_func
            ) for weight, bias, act_func in zip(weights, biases, act_funcs)]
            )

        return cls(layers, loss=LOSSES[loss]())
    
    def __post_init__(self):
        ## ネットワークの損失関数を設定 ##
        if self.loss is None:
            # 特に指定されなければ、出力層の活性化関数に対応する損失関数を選ぶ(最尤推定)
            self.loss = self[-1].h.loss()
        # 損失関数オブジェクトにselfをひもづける
        self.loss.net = self
        # 正準連結関数か否か
        self[-1].h.is_canonical = (type(self.loss) == self[-1].h.loss)
        if not self[-1].h.is_canonical:
            warnings.warn(
                "You are using a non-canonical link function as the activation function of output layer."
            )

        ## ネットワークの各層のニューロン数 ##
        self.shape = tuple(l.size for l in self)

        ## 層間の連結リストとしての構造を構築する ##
        for l in range(1, len(self)):
            # 0層目と1層目, 1層目と2層目, ..., L-1層目とL層目とを連結する
            self[l].prev = self[l-1]
            self[l-1].next = self[l]

        ## 入力層の不要なメンバ変数はNoneにする ##
        self[0].W = self[0].b = self[0].u = self[0].delta = self[0].h = self[0].dJdW = self[0].dJdb = None

    def get_params(self):
        """入力層を除く各層のパラメータのコピーと各層の活性化関数名および損失関数名を取得する.
        """
        return dict(
            weights = [layer.W.copy() for layer in self[1:]],
            biases = [layer.b.copy() for layer in self[1:]],
            act_funcs = [layer.h.copy() for layer in self[1:]],
            loss = self.loss.__class__.__name__
        )

    def copy(self) -> 'mlp':
        return self.__class__.from_params(**self.get_params())

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, x) -> np.ndarray:
        x = check_twodim(x)
        self.forward_prop(x)
        return self[-1].z.copy()
    
    def forward_prop(self, x:np.ndarray) -> None:
        """順伝播. xは入力ベクトル. ミニバッチでもOK"""
        self[0].z = x
        return self[-1].prop_z()
            
    def back_prop(self, t:np.ndarray) -> None:
        """教師信号ベクトルtをもとに誤差逆伝播法による学習を行う."""
        self[-1].delta = self.loss.error(t)  # 出力層の誤差はすぐに計算できる.
        self[1].prop_delta()                 # それを使って誤差を順次前の層へ逆伝播させていく.

    def set_gradient(self) -> None:
        """順伝播と逆伝播により出力zと誤差deltaが計算済みであることを前提に、
        パラメータに関する損失関数の勾配を計算する. 
        """
        for layer in self[1:]:
            layer.set_gradient()
            
    def train(
            self,
            X_train:np.ndarray,
            T_train:np.ndarray,
            eta0:float=0.005,
            optimizer='AdaGrad',
            max_epoch:int=100,
            batch_size=1,
            lamb=0.0001, 
            *args, **kwargs
    ) -> None:
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
        """
        # 途中経過の記録
        self.log = logger(
            net=self,
            n_sample=len(X_train),
            batch_size=batch_size,
            X_train=X_train, T_train=T_train,
            *args, **kwargs
        )
        # パラメータ更新器
        optimizer = OPTIMIZERS[optimizer](net=self, eta0=eta0, lamb=lamb)
        
        try:            
            for epoch in range(max_epoch):
                for X_mini, T_mini in minibatch_iter(X_train, T_train, batch_size):
                    self.forward_prop(X_mini)    # 順伝播
                    self.back_prop(T_mini)       # 逆伝播
                    self.set_gradient()          # パラメータに関する損失の勾配を求める
                    optimizer.update()           # 勾配法による重み更新
                    self.log()                   # ログ出力
                    
        except KeyboardInterrupt:
            warnings.warn('Training stopped by user.')
            
        except NoImprovement as e:
            print(e)
            
        finally:
            self.log.end()
    
    def test(self, X:np.ndarray, T:np.ndarray, log:bool=True) -> None:
        """テストデータ集合(X: 入力, T: 出力)を用いて性能を試験し、正解率を計算する.
        selfは分類器と仮定する. 
        """
        # 多クラス分類問題
        if self[-1].size > 1:
            predicted = np.argmax(self(X), axis=1)
            true      = np.argmax(T, axis=1)# np.dot(T, np.arange(self[-1].size))# np.argmax(T, axis=1)
        # 2クラス分類問題
        else:
            predicted = np.where(self(X) > 0.5, 1, 0)
            true      = T
        
        n_correct = np.count_nonzero(predicted == true)    # 正解サンプル数
        n_sample = len(X)                                  # 全サンプル数
        accuracy = n_correct / n_sample * 100              # 正解率[%]
        print(f"Accuracy = {accuracy:.4f} %")
        if log:
            if self.log is not None:
                self.log.accuracy = accuracy
            else:
                warnings.warn("Can't write log because self.log is None.")



class dropout_layer(layer):
    
    def __init__(self, W=None, b=None, h=None, size=None, first=False, ratio=0.5):
        super().__init__(W=W, b=b, h=h, size=size, first=first)
        self.ratio = ratio

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.size} neurons with {self.h.__class__.__name__} activation, dropout ratio={self.ratio}>"
    
    @classmethod
    def from_layer(cls, layer, ratio=0.5, first=False):
        if first:
            return cls(size=layer.size, first=first, ratio=ratio)
        else:
            return cls(W=layer.W, b=layer.b, h=layer.h, ratio=ratio)

    def make_mask(self, n_input):
        mask = (np.random.rand(1, self.size) >= self.ratio)
        self.mask = np.vstack([mask for _ in range(n_input)])

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
        self.z = super().fire(input)
        self.u *= self.mask
        self.z *= self.mask
        return self.z

    def calc_delta(self, next_delta) -> np.ndarray:
        """次の層における誤差から今の層の誤差を求める.
        """
        self.delta = super().calc_delta(next_delta)
        self.delta *= self.mask
        return self.delta

    
        
@dataclasses.dataclass
class dropout_mlp(mlp):
    ratio: float      = dataclasses.field(default=0.5, repr=False)

    @classmethod
    def from_mlp(cls, net, ratio=0.5):
        # make sure ratio is a list
        if not hasattr(ratio, '__iter__'):
            ratio = [ratio for _ in range(len(net)-1)]
        cp = net.copy()
        layers = [dropout_layer.from_layer(cp[0], first=True, ratio=ratio[0])]
        for l in range(1, len(cp)-1):
            layers.append(dropout_layer.from_layer(cp[l], ratio=ratio[l]))
        layers.append(cp[-1])
        return cls(layers, loss=cp.loss, ratio=ratio)

    def make_mask(self, n_input):
        for layer in self[:-1]:
            layer.make_mask(n_input)
    
    def forward_prop(self, x:np.ndarray) -> None:
        self.make_mask(x.shape[0])
        super().forward_prop(x * self[0].mask)
        


            
# @dataclasses.dataclass
# class ensemble_mlp:
#     mlps: Sequence[mlp] = dataclasses.field(default=None)
#     hard: bool          = dataclasses.field(default=True)
#     soft: bool          = dataclasses.field(default=False)
#     pass


            
class minibatch_iter:
    def __init__(self, X: np.ndarray, T: np.ndarray, batch_size: int, shuffle: bool=True):
        if len(X) != len(T):
            raise ValueError("'X' and 'T' must have the same length.")
        X = check_twodim(X)
        T = check_twodim(T)
        self.n_sample = len(X)
        self.batch_size = batch_size
        if shuffle:
            shuffle_idx = np.random.permutation(self.n_sample)
            self.X = X[shuffle_idx]
            self.T = T[shuffle_idx]
        else:
            # shuffle=Falseはあくまでshuffleの必要性を見るためのもので, 推奨されない.
            # shuffle=Falseとすると、学習中の損失の推移を表すグラフが異様に滑らかになる. 
            self.X = X
            self.T = T

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
    def __init__(self, net, eta0, lamb):
        self.net = net
        self.eta0 = eta0
        self.eta = self.eta0
        n_layer = len(self.net)
        if hasattr(lamb, '__iter__'):
            self.lamb = np.array(lamb)
            if len(self.lamb) != n_layer - 1:
                raise ValueError(
                    "len(lamb) must be = (n_layer - 1)"
                    " in layer-by-layer specification with iterable 'lamb'."
                )
        else:
            self.lamb = np.array([lamb for _ in range(1, n_layer)])
            
        if np.any(self.lamb):
            self.regularization = True
        else:
            self.regularization = False
        # パラメータの1ステップあたり更新量
        self.dWs = [None] + [0 for _ in range(1, len(self.net))] # 重み行列用
        self.dbs = [None] + [0 for _ in range(1, len(self.net))] # バイアス用

    def get_update(self):
        """Set new values to`dWs` and `dbs`. Override this method when you create a new subclasses.
        """
        raise NotImplementedError

    def add_regularization_term(self):
        for layer, lamb in zip(self.net[1:], self.lamb):
            layer.dJdW += lamb * layer.W
            # layer.dJdb += lamb * layer.b # バイアスには適用しない
            
    def update(self):
        if self.regularization:
            self.add_regularization_term()
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
    def __init__(self, net, eta0, lamb, eps=1e-7):
        super().__init__(net, eta0, lamb)
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
    def __init__(self, net, eta0, lamb, momentum=0.9):
        super().__init__(net, eta0, lamb)
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
    param : float            = dataclasses.field(default=1.0)
    # 出力層の活性化関数として用いた場合の対応する損失関数クラス
    loss : Type["loss_func"] = dataclasses.field(init=False, default=None, repr=False)
    # 正準連結関数かどうか
    is_canonical : bool      = dataclasses.field(default=None)

    def __call__(self, u):
        """活性化関数の値h(u)そのもの."""
        raise NotImplementedError

    def val2deriv(self, z):
        """活性化関数の関数値z = h(u)の関数として導関数h\'(u)の値を計算する."""
        raise NotImplementedError

    def copy(self):
        return self.__class__(param=self.param, is_canonical=self.is_canonical)

    @staticmethod
    def make(arg):
        if isinstance(arg, str):
            return ACTIVATIONS[arg]()
        elif isinstance(arg, act_func):
            return arg
        else:
            raise TypeError(
                "'arg' must be either str(name of act_func) or act_func object,"
                f" not {type(arg)}"
            )



@dataclasses.dataclass
class linear(act_func):
    """線形関数"""
    def __post_init__(self):
        self.loss = mean_square

    def __call__(self, u):
        return self.param * u
    
    def val2deriv(self, z):
        return self.param


@dataclasses.dataclass
class step(act_func):
    """ヘヴィサイドのステップ関数"""
    def __post_init__(self):
        # sigmoid(a*u)でa -> +inf とした極限だと思えば、交差エントロピーでいいのでは??
        self.loss = cross_entropy
        
    def __call__(self, u):
        return np.maximum(np.sign(u), 0.0)
    
    def val2deriv(self, z):
        return 0.0


@dataclasses.dataclass
class sigmoid(act_func):
    # val2deriv計算時の桁落ち防止に使うつもりだったが、必要ないかも
    eps: float = dataclasses.field(default=1e-4)
    
    """シグモイド関数"""
    def __post_init__(self):
        self.loss = cross_entropy
        
    def __call__(self, u):
        return 0.5 * (1.0 + np.tanh(0.5 * self.param * u))
    
    def val2deriv(self, z):
        return self.param * z * (1.0 - z) # np.maximum(z, self.eps) * (1.0 - np.minimum(z, 1-self.eps))


@dataclasses.dataclass
class ReLU(act_func):
    """ReLU"""
    def __call__(self, u):
        return np.maximum(self.param * u, 0.0)
    
    def val2deriv(self, z):
        return np.where(z > 0.0, self.param, 0.0)


@dataclasses.dataclass
class LeakyReLU(act_func):
    """Leaky ReLU"""
    alpha:float = dataclasses.field(default=0.01)
        
    def __call__(self, u):
        return np.maximum(self.param * u, self.alpha * u)
    
    def val2deriv(self, z):
        return np.where(z > 0, self.param, self.alpha)


@dataclasses.dataclass
class softmax(act_func):
    """ソフトマックス関数"""
    def __post_init__(self):
        self.loss = mul_cross_entropy
        
    def __call__(self, u):
        tmp = u - u.max()
        return np.exp(tmp) / np.sum(np.exp(tmp), axis=1, keepdims=True)
    
    def val2deriv(self, z):
        return np.array([np.diag(z[i]) - np.outer(z[i], z[i]) for i in range(len(z))])
        # raise Exception("出力層の活性化関数の微分は使わないはず")


ACTIVATIONS = {
    'identity'   : linear,
    'linear'     : linear,
    'step'       : step,
    'threshold'  : step,
    'sigmoid'    : sigmoid,
    'logistic'   : sigmoid,
    'relu'       : ReLU,
    'ReLU'       : ReLU,
    'LeakyReLU'  : LeakyReLU,
    'leakyrelu'  : LeakyReLU,
    'softmax'    : softmax
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
        last_layer = self.net[-1]
        delta = last_layer.z - t
        # mean_squareとsigmoid/softmaxの組み合わせならこれで正しい。ほかの組み合わせでもこれでいいのかは未確認(PRMLのpp.211?まだきちんと追ってない)!!!
        if not last_layer.h.is_canonical:
            if isinstance(last_layer.h, softmax):
                delta = np.matmul(delta.reshape((delta.shape[0], 1, delta.shape[1])), last_layer.h.val2deriv(last_layer.z))
                delta = delta.reshape((delta.shape[0], -1))
            else:
                delta *= last_layer.h.val2deriv(last_layer.z)
        return delta


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


LOSSES = {
    'mean_square'       : mean_square,
    'cross_entropy'     : cross_entropy,
    'mul_cross_entropy' : mul_cross_entropy,
    None                : lambda: None
}



class NoImprovement(Exception):
    """Raised when no progress is being made any more in training."""
    pass



@dataclasses.dataclass
class logger:
    """学習経過の記録と損失のグラフの描画, および早期終了の制御"""
    net : mlp                  = dataclasses.field(default=None, repr=False)
    iterations : int           = dataclasses.field(default=0)
    loss: Sequence[float]      = dataclasses.field(default_factory=list)
    count: Sequence[int]       = dataclasses.field(default_factory=list)
    delta_iter: int            = dataclasses.field(default=None)
    n_sample : int             = dataclasses.field(default=None)
    batch_size: int            = dataclasses.field(default=None)
    X_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    T_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    X_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    T_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    _compute_val_loss : bool   = dataclasses.field(default=False, repr=False)
    val_loss: Sequence[float]  = dataclasses.field(default_factory=list)
    color : str                = dataclasses.field(default='tab:blue', repr=False)
    color2 : str               = dataclasses.field(default='tab:orange', repr=False)
    how: str                   = dataclasses.field(default='plot', repr=False)
    delta_epoch : int          = dataclasses.field(default=20, repr=False)
    early_stopping: bool       = dataclasses.field(default=True)
    patience_epoch: int        = dataclasses.field(default=8)
    _no_improvement_iter: int  = dataclasses.field(init=False, default=0, repr=False)
    tol: float                 = dataclasses.field(default=1e-4)
    best_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    best_params_val: dict      = dataclasses.field(init=False, default=None, repr=False)
    stop_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    should_stop_iter : int     = dataclasses.field(init=False, default=None)
    AIC : float                = dataclasses.field(init=False, default=None)
    BIC : float                = dataclasses.field(init=False, default=None)
    accuracy: float            = dataclasses.field(default=None)
    time: float                = dataclasses.field(default=None)

    def __post_init__(self):
        # 検証用データ(X_val, T_val)に対する誤差も計算するかどうか
        if not((self.X_val is None) and (self.T_val is None)):
            self.X_val = check_twodim(self.X_val)
            self.T_val = check_twodim(self.T_val)
            self._compute_val_loss = True
        # 1エポックあたりiteration数
        self._iter_per_epoch = int(np.ceil(self.n_sample / self.batch_size))
        # 記録をとる頻度はどんなに粗くても1エポック
        if self.delta_iter is None:
            self.delta_iter = self._iter_per_epoch
        self.delta_iter = min(self.delta_iter, self._iter_per_epoch)
        # 損失の変化をどう表示するか
        if self.how == 'both':
            self._plot, self._stdout = True, True
        elif self.how == 'plot':
            self._plot, self._stdout = True, False
        elif self.how == 'stdout':
            self._plot, self._stdout = False, True
        elif self.how == 'off':
            self._plot, self._stdout = False, False
        else:
            raise ValueError("logger.how must be either of the followings: 'both', 'plot', 'stdout' or 'off'")

        # 損失のグラフをリアルタイムで
        if self._plot:
            self.fig, self.ax, self.secax = self.init_plot()
            plt.ion()

        # 損失がself.tol以上改善しなければ学習を打ち切る
        self.best_loss = np.inf
        if self._compute_val_loss:
            self.best_val_loss = np.inf

        # 学習開始時間を記録
        self._t0 = time.time()

    def init_plot(self):
        fig, ax = plt.subplots(constrained_layout=True)
        
        ax.set(xlabel="iterations", ylabel="loss")
        secax = ax.secondary_xaxis(
            'top',
            functions=(lambda count: count/self._iter_per_epoch,
                       lambda epoch: epoch*self._iter_per_epoch)
        )
        
        secax.xaxis.set_major_locator(ticker.AutoLocator())
        secax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        secax.set_xlabel('epochs')
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=None)
        ax.grid(axis='y', linestyle='--')

        return fig, ax, secax

    def __call__(self) -> None:
        if self.iterations % self.delta_iter == 0:
            # log出力
            T_mini = self.net[-1].z - self.net[-1].delta
            self.loss.append(self.net.loss(None, T_mini))
            self.count.append(self.iterations)
            # 現在のエポック
            epoch = self.iterations // self._iter_per_epoch
            # epoch内で何番目のiterationか
            idx_iter = self.iterations % self._iter_per_epoch
            # epoch内で何番目のパターンか
            idx_sample = min(
                (idx_iter + 1) * self.batch_size,
                self.n_sample
            )
            if self._compute_val_loss:
                self.val_loss.append(self.net.loss(self.X_val, self.T_val))
            
            logstr = f"Epoch {epoch:3}, Pattern {idx_sample:5}/{self.n_sample}: Loss = {self.loss[-1]:.3e}"
            if self._compute_val_loss:
                logstr += f" (training), {self.val_loss[-1]:.3e} (validation)"
            
            if self._stdout:
                print(logstr)

            if self._plot:
                self.ax.set_xlim(0, (int(np.ceil((epoch+1e-4)/self.delta_epoch))*self.delta_epoch) * self._iter_per_epoch)
                
                if self._compute_val_loss:
                    self.ax.plot(self.count[-2:], self.val_loss[-2:], c=self.color2)
                    if len(self.count) >= 2:
                        marker = '^' if self.val_loss[-2] < self.val_loss[-1] else 'v'
                        self.ax.scatter(self.count[-1:], self.val_loss[-1:],
                                        marker=marker, fc=self.color2, ec='k')

                self.ax.plot(self.count[-2:], self.loss[-2:], c=self.color)
                self.ax.set_title(logstr, fontsize=10)

                plt.show()
                plt.pause(0.1)

            # 早期終了など
            last_loss = self.avrg_last_epoch(self.loss)
            
            if self._compute_val_loss:
                last_val_loss = self.avrg_last_epoch(self.val_loss)
                # 検証用データに対する損失に一定以上の改善が見られない場合
                if last_val_loss > (self.best_val_loss - self.tol):
                    self._no_improvement_iter += self.delta_iter
                else:
                    self._no_improvement_iter = 0
                # 現時点までの暫定最適値を更新(検証用データ) 
                if last_val_loss < self.best_val_loss:
                    self.best_val_loss = self.val_loss[-1]
                    self.best_params_val = self.net.get_params()
                    
            else:
                # 訓練データに対する損失に一定以上の改善が見られない場合
                if last_loss > (self.best_loss - self.tol):
                    self._no_improvement_iter += self.delta_iter
                else:
                    self._no_improvement_iter = 0
            
            # 現時点までの暫定最適値を更新(訓練データ)
            if last_loss < self.best_loss:
                self.best_loss = last_loss
                self.best_params = self.net.get_params()

            _no_improvement_epoch = self._no_improvement_iter / self._iter_per_epoch
            if _no_improvement_epoch > self.patience_epoch:
                self.should_stop_iter = self.iterations
                which = 'Validation' if self._compute_val_loss else 'Training'
                no_improvement_msg = (
                    f"{which} loss did not improve more than "
                    f"tol={self.tol} for the last {self.patience_epoch} epochs"
                    f" ({self.should_stop_iter} iterations &"
                    f" {self.should_stop_iter / self._iter_per_epoch:.1f} epochs so far)."
                )
                if self.early_stopping:
                    self.stop_params_val = self.net.get_params()
                    raise NoImprovement(no_improvement_msg)
                self.stop_params = self.net.get_params()
                warnings.warn(no_improvement_msg)

        self.iterations += 1

    def avrg_last_epoch(self, loss_list):
        """直近1エポックでの損失の平均."""
        last_epoch = loss_list[-int(self._iter_per_epoch / self.delta_iter + 0.5):]
        avrg_loss = np.mean(last_epoch)
        return avrg_loss
        
    def end(self) -> None:
        # record time elapsed
        self._tf = time.time()
        self.time = self._tf - self._t0
        # calculate AIC & BIC (This is correct only when using (sigmoid, cross_entropy) or (softmax, mul_cross_entropy).)
        net = self.net
        if (isinstance(net.loss, cross_entropy)
            and
            net[-1].h.is_canonical):
            nll = net.loss(self.X_train, self.T_train) * self.n_sample # negative log likelihood
            n_param = 0                                                # number of parameters in net
            for layer in net[1:]:
                n_param += layer.W.size + layer.b.size
            self.AIC = 2 * nll + 2 * n_param
            self.BIC = 2 * nll + n_param * np.log(self.n_sample)
            
    
    def show(self, color='tab:blue', color2='tab:orange', *args, **kwargs):
        """学習終了後にプロット(なぜかlinesが表示されない。原因不明)
        """
        fig, ax, secax = self.init_plot()
        ax.plot(self.count, self.loss, color=color, *args, **kwargs)
        if self.val_loss:
            ax.plot(self.count, self.val_loss, color=color2, *args, **kwargs)
        return fig, ax, secax

    def to_file(self, fname) -> None:
        with open(fname, "w") as f:
            f.write(repr(self))
            
    @classmethod
    def from_file(cls, fname):
        with open(fname) as f:
            return eval(f.read())


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
