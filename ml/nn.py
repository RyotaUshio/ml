"""Multi-layer Perceptron.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import dataclasses
from typing import Type, Sequence, List, Callable
import warnings
import pickle
import copy

from . import base, utils
from .exceptions import NoImprovement



class layer:
    """A layer in multi-layer perceptron.
    
    Parameters
    ----------
    W : array_like, default=None
        The weight matrix between the layer and the previous one.
        The (i, j) element represents the weight value between the i-th neuron
        of the previous layer and the j-th in this layer. If `first=True`, 
        this parameter is ignored.
    b : array_like, default=None
        The bias vector. If `first=True`, this parameter is ignored.
    h : str or act_func, defalut=None
        The activation function. If `str` is passed, it is iterpreted as 
        the name of one. If `first=True`, this parameter is ignored.
    size : int, default=None
        Number of neurons contained in the layer. It is necessary only if 
        `first == True`. Otherwise, it is inferred from the shapes of `W`
        and `b`.
    first : bool, default=None
        If true, the layer is treated as an input layer of a network, and 
        parameters except for `size` will be ignored. In this case, `size` 
        must be specified because it cannot be inferred from other argments.

    Attributes
    ----------
    W : np.ndarray
        The weight matrix between the layer and the previous one.
    b : np.ndarray
        The bias vector.
    h : act_func
        The activation function.
    size : int
        Number of neurons contained in the layer.
    u : np.ndarray
        `u = x @ W + b`, where `x` denotes the input vector.
    z : np.ndarray
        The output vector of the layer, computed as `z = h(u)`.
    delta : np.ndarray
        The error vector of the layer.
    dJdW : np.ndarray
        Gradient of loss with repect to the weight matrix.
    dJdb : np.ndarray
        Gradient of loss with repect to the bias vector.
    prev : layer or None
        The previous layer in the network.
    next : layer or None
        The next layer in the network.
    """
    
    def __init__(self, W:Sequence=None, b:Sequence=None, h:'act_func'=None, size:int=None, first:bool=False):
        # for an non-input layer
        if not first:
            # make sure that all matrices and vectors are a two-dimensional numpy.ndarray
            self.W = utils.check_twodim(W)
            self.b = utils.check_twodim(b)
            # validation of the shape of weight matrix and bias vector
            if self.W.shape[1] != self.b.shape[1]:
                raise ValueError(
                    "Expected self.W.shape[1] == self.b.shape[1],"
                    f"got self.W of shape {self.W.shape} and "
                    f"self.b of shape {self.b.shape}"
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
                    "The size of an input layer must be specified"
                    " because it cannot be inferred from W.shape or b.shape"
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
        """Whether `self` is an input layer of a network or not.
        """
        return (self.prev is None) and (self.next is not None)
    
    def is_last(self) -> bool:
        """Whether `self` is an output layer of a network or not.
        """
        return (self.prev is not None) and (self.next is None)

    def is_hidden(self) -> bool:
        """Whether `self` is a hidden layer of a network or not.
        """
        return (self.prev is not None) and (self.next is not None)

    def is_unconnected(self) -> bool:
        return (self.prev is None) and (self.next is None)

    def is_connected(self) -> bool:
        """Whether `self` is connected to other layers or not.
        """
        return not self.is_unconnected()

    def fire(self) -> None:
        """Make the neurons in the layer activated.
        """
        self.u = self.prev.z @ self.W + self.b
        self.z = self.h( self.u )
        
    def prop_z(self) -> None:
        """入力層が現在保持している信号zを使って、ネットワークの入力層から
        自分自身まで信号を順伝播させる.
        """
        if self.is_first():
            pass
        else:
            self.prev.prop_z()
            self.fire()

    def calc_delta(self) -> None:
        """次の層における誤差から今の層の誤差を求める.
        """
        self.delta = self.h.val2deriv(self.z) * (self.next.delta @ self.next.W.T)
    
    def prop_delta(self) -> None:
        """出力層が現在保持している誤差deltaを使って、ネットワークの出力層から
        自分自身まで誤差を逆伝播させる.
        """
        if self.is_last():
            pass
        else:
            self.next.prop_delta()
            self.calc_delta()
        
    def set_gradient(self) -> None:
        """順伝播と逆伝播により出力zと誤差deltaが計算済みであることを前提に、
        パラメータに関する損失関数の勾配を計算する. 
        """
        batch_size = self.z.shape[0]
        self.dJdW = (self.prev.z.T @ self.delta) / batch_size
        self.dJdb = np.mean(self.delta, axis=0, keepdims=True)

    def set_input(self, x : np.ndarray) -> None:
        self.z = x
        
        
        
class _dropout_layer_base(layer):
    """Dropout layer.

    Dropout is an effective method to avoid overfitting.

    Attributes
    ----------
    ratio : float
        The probability that each neuron is dropped out.

    References
    ----------
    https://github.com/chainer/chainer/blob/eddf10e4af3756dbf32149d0b6ad91cebcf529c1/chainer/functions/noise/dropout.py
    """
    
    def __init__(self, W=None, b=None, h=None, size=None, first=False, ratio=0.5):
        super().__init__(W=W, b=b, h=h, size=size, first=first)
        if not (0.0 <= ratio < 1):
            raise ValueError("'ratio' must be 0 <= ratio < 1")
        self.ratio = ratio
        self._now_training = False

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.size} neurons with {self.h.__class__.__name__} activation, dropout ratio={self.ratio}>"
    
    @classmethod
    def from_layer(cls, layer, ratio=0.5, first=False) -> Type['_dropout_layer_base']:
        """Convert a `layer` into a `dropout_layer`.
        """
        if first:
            return cls(size=layer.size, first=first, ratio=ratio)
        else:
            return cls(W=layer.W, b=layer.b, h=layer.h, ratio=ratio)

    def make_mask(self) -> None:
        self.mask = np.random.rand(self.size) >= self.ratio

    def fire(self) -> None:
        super().fire()
        if self._now_training:
            self._fire_train()
        else:
            self._fire_test()

    def _fire_train(self) -> None:
        self.make_mask()
        self.z *= self.mask

    def _fire_test(self) -> None:
        raise NotImplementedError
        
    def calc_delta(self) -> None:
        super().calc_delta()
        self.delta *= self.mask

    def set_input(self, x : np.ndarray) -> None:
        if self._now_training:
            self.make_mask()
            self.z = x * self.mask
        else:
            self.z = x * (1 - self.ratio)

      
class dropout_layer(_dropout_layer_base):
    def _fire_test(self) -> None:
        self.z *= 1 - self.ratio
    

class inverted_dropout_layer(_dropout_layer_base):
    def make_mask(self) -> None:
        super().make_mask()
        self.mask = self.mask / (1 - self.ratio)

    def _fire_test(self) -> None:
        pass


    
class conv_layer(layer):
    """A convolutional layer class.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    
class pool_layer(layer):
    """A pooling layer class.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


    
@dataclasses.dataclass
class mlp(base._estimator_base):
    """Base class for MLP: multi-layer perceptron.

    Parameters
    ----------
    layers : Sequence of layer
        A sequence of layer objects contained in the network.
    loss : loss_func, default=None
        The loss function which will be used in network training process with 
        back propagation. If not given, the one is chosen that corresponds to
        the last layer's activation function in the sense of maximum likeli-
        -hood estimation.
    
    Attributes
    ----------
    layers : Sequence of layer
        A sequence of `layer` objects contained in the network.
    loss : loss_func
        The loss function which will be used in network training process with 
        back propagation.
    shape : tuple
        The number of neurons contained in each layer of the network, that is,
        a tuple of (n_unit_1st_layer, n_unit_2nd_layer, ..., n_unit_last_layer).
    log : logger
        A `logger` object, which records various information, including values 
        of loss function at each epoch in training, AIC and BIC and so 
        on. It also controls the whole process of early stopping. For more 
        details, see `logger`'s doc.
    dropout : bool
        Whether to use Dropout or not.
    """
    
    layers: Sequence[layer]
    loss: 'loss_func' = None
    log: 'logger' = dataclasses.field(init=False, default=None, repr=False)
    dropout : bool = dataclasses.field(init=False, default=False, repr=False)
    dropout_ratio : dataclasses.InitVar[Sequence[float]] = None
    inverted : dataclasses.InitVar[bool] = True

    @classmethod
    def from_shape(
            cls, shape: Sequence[int],
            act_funcs: Sequence[str]=None, *,
            hidden_act=None, out_act=None,
            loss=None,
            sigmas=None,
            **kwargs
    ) -> Type['mlp']:
        """Create a new `mlp` object with specified shape & activation functions.

        Parameters
        ----------
        shape : Sequence of int
            The number of neurons contained in each layer (i.e. (n_neuron of the 
            1st layer, n_neuron of the 2nd layer, ..., n_neuron of the last layer)).
        act_funcs : Sequence of str or act_func
            The activation functions of each layer. 
            ex) For a MLP with 2 hidden layers: [None, 'ReLU', 'sigmoid', 'softmax']
        hidden_act, out_act : str or act_func, optional
            The activation functions can be specified with these parameters, 
            instead of `act_funcs`.
        loss : str or loss_func, optional
            The loss function used in network training with back-propagation. If 
            not given, it will be set according to the last layer's activation function.
        sigmas : Sequence of float or float, optional
            Stddev of Gaussian distribution from which weights in each layer are 
            sampled. If not given, they are set in accordance with the methods 
            proposed by Xavier[1]_ & He[2]_.
        dropout_ratio : Sequence of float or float, optional
            If given, the input & hidden layers are converted into `dropout_layer` 
            according to specified dropout ratios.

        Returns
        -------
        A new `mlp` object.
        
        References
        ----------
        .. [1] Xavier Glorot, Yoshua Bengio. "Understanding the difficulty of 
           training deep feedforward neural networks." AISTATS 2010.
        .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Delving Deep
           into Rectifiers: Surpassing Human-Level Performance on ImageNet 
           Classification." arXiv:1502.01852. 2015.
        """
        n_layer = len(shape)

        # make sure sigmas is a list
        if not hasattr(sigmas, '__iter__'):
            sigmas = [sigmas for _ in range(n_layer)]
        sigmas = list(sigmas)

        # settting activation functions
        act_funcs = cls._make_act_funcs(shape, act_funcs, hidden_act, out_act)
        
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

        net = cls(layers, loss=LOSSES[loss](), **kwargs)
        return net

    @staticmethod
    def _make_act_funcs(shape, act_funcs, hidden_act, out_act) -> List['act_func']:
        if act_funcs is not None:
            return act_funcs
        
        # 中間層の活性化関数
        if isinstance(hidden_act, (str, act_func)):
            hidden_act = [hidden_act for _ in range(len(shape[1:-1]))]
        elif hasattr(hidden_act, '__iter__'):
            hidden_act = list(hidden_act)
        else:
            raise TypeError("'hidden act' of an invalid type was passed.")
        if len(hidden_act) != len(shape[1:-1]):
            raise ValueError("Incompatible length: 'shape' & 'hidden_act'")
    
        return [None] + hidden_act + [out_act]

    @classmethod
    def from_params(
            cls, weights:Sequence,
            biases:Sequence,
            act_funcs:Sequence[str],
            loss=None,
            include_first=False,
            *args, **kwargs
    ):
        """Make a mlp object by specifying paramters (= weight matrices and bias
        vectors) and activation functions of each layer.
        """
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

        return cls(layers, loss=LOSSES[loss](), *args, **kwargs)
    
    def __post_init__(self, dropout_ratio, inverted):
        ## Dropoutの設定 ##
        if dropout_ratio:
            self.set_dropout(dropout_ratio, inverted)
            self.dropout = True
        
        ## ネットワークの損失関数を設定 ##
        if self.loss is None:
            # 特に指定されなければ、出力層の活性化関数に対応する損失関数を選ぶ(最尤推定)
            self.loss = self[-1].h.loss_type()
        # 損失関数オブジェクトにselfをひもづける
        self.loss.net = self
        # 正準連結関数か否か
        self[-1].h.is_canonical = (type(self.loss) == self[-1].h.loss_type)
        if not self[-1].h.is_canonical:
            warnings.warn(
                "You are using a non-canonical link function as the activation function of output layer."
            )

        ## ネットワークの各層のニューロン数 ##
        self.shape = tuple(l.size for l in self)

        ## 層間の連結リストとしての構造を構築する ##
        self._connect_layers()

        ## 入力層の不要なメンバ変数はNoneにする ##
        self[0].W = self[0].b = self[0].u = self[0].delta = self[0].h = self[0].dJdW = self[0].dJdb = None

    def _connect_layers(self):
        for l in range(1, len(self)):
            # 0層目と1層目, 1層目と2層目, ..., L-1層目とL層目とを連結する
            self[l].prev = self[l-1]
            self[l-1].next = self[l]

    def get_params(self):
        """入力層を除く各層のパラメータのコピーと各層の活性化関数名および損失関数名を取得する.
        """
        params =  dict(
            weights = [layer.W.copy() for layer in self[1:]],
            biases = [layer.b.copy() for layer in self[1:]],
            act_funcs = [layer.h.copy() for layer in self[1:]],
            loss = self.loss.__class__.__name__,
        )
        if self.dropout:
            params.update(dict(
                dropout_ratio = [layer.ratio for layer in self[:-1]],
                inverted = isinstance(self[0], inverted_dropout_layer)
            ))
        return params
            

    def copy(self, all=False) -> 'mlp':
        """Make a copy of the network.

        Parameters
        ----------
        all : bool, default=False
            If False (default), only parameters which are necessary for 
            inference, such as layer.W, layer.b, layer.h.
            If True, all the internal variables such as layer.z, layer.u, 
            layer.delta, ... are also copied. This takes a little bit of time.

        Returns
        -------
        A copied object.
        """
        if all:
            return copy.deepcopy(self)
        return self.__class__.from_params(**self.get_params())

    def save(self, filename):
        """Save the network as a pickle.

        This method is an alias for utils.save.
        """
        utils.save(self, filename)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.layers[key]

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, x) -> np.ndarray:
        x = utils.check_twodim(x)
        self.forward_prop(x)
        return self[-1].z.copy()

    def forward_prop(self, x : np.ndarray) -> None:
        """Forward propagation computation.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_sample, n_feature)
            An input signal of the network.
        """
        self[0].set_input(x)
        self[-1].prop_z()
            
    def back_prop(self, t : np.ndarray) -> None:
        """Compute errors in each layer with backpropagation.
        
        Parameters
        ----------
        t : nd.ndarray of shape (n_sample, n_target)
            A target signal.
        """
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
            T_train:np.ndarray, *,
            eta0:float=1e-2,
            optimizer='Momentum',
            max_epoch:int=100,
            batch_size=200,
            lamb=1e-4,
            **kwargs
    ) -> None:
        """
        訓練データ集合(X: 入力, T: 出力)をネットワークに学習させる. エポック数がmax_iterを超えるか、
        シグナルSIGINT(Ctrl+C)が送出されると学習を打ち切る。

        Parameters
        ----------
        X_train : np.ndarray of shape (n_sample, n_feature)
            訓練データセットの入力群.
        T_train : np.ndarray of shape (n_sample, n_target)
            訓練データセットの出力群.
        eta0: float, default=0.05
            学習係数(の初期値)を表すパラメータ.
        optimizer: {'SGD', 'AdaGrad', 'Momentum'(, 'Adam', 'LBFGS')}, default='AdaGrad'
            学習に用いる最適化手法.
        max_epoch : int, default=100
            最大反復エポック数.
        batch_size : int, default=200
            ミニバッチ学習に用いるミニバッチのサイズ. 
        lamb : float, default=0.0001
            Coeffiecient of weight decay.
        *args, **kwargs
            Other arguments passed to the constructor of `logger`.

        Returns
        -------
        None
        """
        # パラメータ更新器
        optimizer = OPTIMIZERS[optimizer](net=self, eta0=eta0, lamb=lamb)
        # 途中経過の記録
        self.log = self.log_init(
            net=self,
            n_sample=len(X_train),
            batch_size=batch_size,
            X_train=X_train, T_train=T_train,
            optimizer=optimizer,
            **kwargs
        )
        if self.dropout:
            self._set_training_flag(True)
        
        try:            
            for epoch in range(max_epoch):
                for X_mini, T_mini in utils.minibatch_iter(X_train, T_train, batch_size):
                    self.train_one_step(X_mini, T_mini, optimizer)

        except KeyboardInterrupt:
            warnings.warn('Training stopped by user.')
            
        except NoImprovement as e:
            print(e)

        self.log.end()
        if self.dropout:
            self._set_training_flag(False)

    def train_one_step(self, X_mini, T_mini, optimizer):
        self.forward_prop(X_mini)    # 順伝播
        self.back_prop(T_mini)       # 逆伝播
        self.set_gradient()          # パラメータに関する損失の勾配を求める
        optimizer.update()           # 勾配法による重み更新
        self.log()                   # ログ出力

    def log_init(self, **kwargs):
        return logger(**kwargs)
        
    @classmethod
    def fit(
            cls,
            X_train:np.ndarray,
            T_train:np.ndarray,
            hidden_shape: Sequence[int],
            act_funcs: Sequence[str]=None, *,
            hidden_act='ReLU', out_act=None,
            loss=None,
            sigmas=None,
            dropout_ratio: Sequence[float] =None,
            inverted=True,
            **kwargs
    ):
        shape = [X_train.shape[1]] + list(hidden_shape) + [T_train.shape[1]]
        net = cls.from_shape(
            shape=shape,
            act_funcs=act_funcs,
            hidden_act=hidden_act,
            out_act=out_act,
            loss=loss, sigmas=sigmas, dropout_ratio=dropout_ratio, inverted=inverted
        )
        net.train(X_train=X_train, T_train=T_train, **kwargs)
        return net

    # ______________________ Dropout methods ______________________
    def set_dropout(self, ratio, inverted):
        # make sure ratio is a list
        if not hasattr(ratio, '__iter__'):
            ratio = [ratio for _ in range(len(self)-1)]
        ratio = list(ratio)

        if len(ratio) != len(self) - 1:
            raise ValueError(
                "`ratio` must be `len(ratio) == len(self) - 1`"
                " because Dropout can be turned on only in an input layer and hidden layers."
            )

        layer_type = inverted_dropout_layer if inverted else dropout_layer
        self.layers[0] = layer_type.from_layer(self[0], first=True, ratio=ratio[0])
        for l in range(1, len(self)-1):
            self.layers[l] = layer_type.from_layer(self[l], ratio=ratio[l])
        self._connect_layers()
            
    def _set_training_flag(self, b: bool):
        for layer in self[:-1]:
            layer._now_training = b



    
@dataclasses.dataclass
class ensemble_mlp(base._estimator_base):
    nets: Sequence[mlp]
    how: str = 'soft'

    def __post_init__(self):
        if self.how == 'hard':
            self._hard = True
        elif self.how == 'soft':
            self._hard = False
        else:
            raise ValueError("'how' must be either 'hard' or 'soft.'")
        self._soft = not self._hard

    def train_all(self, *args, **kwargs):
        for net in nets:
            net.train(*args, **kwargs)
    
    def hard_ensemble(self, x):
        agg = 0.0
        for net in self.nets:
            agg += net.predict_one_of_K(x)
        return agg

    def soft_ensemble(self, x):
        agg = 0.0
        for net in self.nets:
            agg += net(x)
        agg /= len(self.nets)
        return agg

    def __call__(self, x):
        if self._hard:
            return self.hard_ensemble(x)
        elif self._soft:
            return self.soft_ensemble(x)
        else:
            raise Exception("Both of '_hard' & '_soft' are set False. Something went wrong.")


    

@dataclasses.dataclass
class mlp_classifier(mlp, base.classifier_mixin):
    """Classification with MLP: multi-layer perceptron.

    Parameters
    ----------
    (This class is supposed to be constructed via `from_shape()`, not 
    by calling `__init__()` directly.)
    
    layers : Sequence of layer
        A sequence of layer objects contained in the network.
    loss : loss_func, default=None
        The loss function which will be used in network training process with 
        back propagation. If not given, the one is chosen that corresponds to
        the last layer's activation function in the sense of maximum likeli-
        -hood estimation.

    Attributes
    ----------
    layers : Sequence of layer
        A sequence of `layer` objects contained in the network.
    loss : loss_func
        The loss function which will be used in network training process with 
        back propagation.
    shape : tuple
        The number of neurons contained in each layer of the network, that is,
        a tuple of (n_unit_1st_layer, n_unit_2nd_layer, ..., n_unit_last_layer).
    log : logger
        An `logger` object, which records various information, including the 
        value of loss function at each epoch in training, AIC and BIC and so 
        on. It also controls the whole process of early stopping. For more 
        details, see `logger`'s doc.
    dropout : bool
        Whether to use Dropout or not.
    classification_type : {'binary', 'multi'}
        The type of classification.
    
    Examples
    --------
    >>> import nn, utils
    >>> (X_train, T_train), (X_test, T_test) = utils.load()
    >>> n_feature, n_target = X_train.shape[1], T_train.shape[1]
    >>> n_hidden_units = 20
    >>> net = nn.mlp_classifier.from_shape(
    ...     [n_feature, n_hidden_units, n_target]
    ... )
    >>> net.train(X_train, T_train)
    >>> net.test(X_test, T_test, verbose=True)
    Accuracy: 97.4300 %
    """
    
    classification_type: str = dataclasses.field(init=False)

    def __post_init__(self, dropout_ratio, inverted):
        super().__post_init__(dropout_ratio, inverted)
        n_output = self.shape[-1]
        if n_output >= 2:
            self.classification_type = 'multi'
        elif n_output == 1:
            self.classification_type = 'binary'

    @classmethod
    def from_shape(cls,
                   shape,
                   hidden_act='ReLU',
                   loss=None,
                   sigmas=None,
                   *args, **kwargs
    ) -> 'mlp_classifier':
        out_act = cls._get_out_act_name(shape[-1])
        kwargs.update({'out_act' : out_act})
        return super().from_shape(
            shape=shape,
            hidden_act=hidden_act,
            loss=loss,
            sigmas=sigmas,
            *args, **kwargs
        )
    
    @classmethod
    def fit(cls,
            X_train:np.ndarray,
            T_train:np.ndarray,
            hidden_shape: Sequence[int],
            hidden_act='ReLU',
            loss=None,
            sigmas=None,
            *args, **kwargs):
        return super().fit(
            X_train=X_train, T_train=T_train, hidden_shape=hidden_shape,
            hidden_act=hidden_act,
            loss=loss, sigmas=sigmas, *args, **kwargs
        )

    @staticmethod
    def _get_out_act_name(n_output):
        # 出力層の活性化関数
        if n_output >= 2:
            out_act = 'softmax'
        elif n_output == 1:
            out_act = 'sigmoid'
        else:
            raise ValueError(f"'n_output' must be a positive integer, not {n_output}.")
        return out_act

    def predict_label(self, x):
        threshold = None
        if self.classification_type == 'binary':
            out_act = self[-1].h
            threshold = 0.5 * (out_act(np.inf) + out_act(-np.inf))
        return super().predict_label(x, threshold=threshold)

    


@dataclasses.dataclass
class mlp_regressor(mlp, base.regressor_mixin):
    """Regression with MLP: multi-layer perceptron.

    Parameters
    ----------
    (This class is supposed to be constructed via `from_shape()`, not 
    by calling `__init__()` directly.)
    
    layers : Sequence of layer
        A sequence of layer objects contained in the network.
    loss : loss_func, default=None
        The loss function which will be used in network training process with 
        back propagation. If not given, the one is chosen that corresponds to
        the last layer's activation function in the sense of maximum likeli-
        -hood estimation.

    Attributes
    ----------
    layers : Sequence of layer
        A sequence of `layer` objects contained in the network.
    loss : loss_func
        The loss function which will be used in network training process with 
        back propagation.
    shape : tuple
        The number of neurons contained in each layer of the network, that is,
        a tuple of (n_unit_1st_layer, n_unit_2nd_layer, ..., n_unit_last_layer).
    log : logger
        An `logger` object, which records various information, including the 
        value of loss function at each epoch in training, AIC and BIC and so 
        on. It also controls the whole process of early stopping. For more 
        details, see `logger`'s doc.
    dropout : bool
        Whether to use Dropout or not.
    """
    
    @classmethod
    def from_shape(cls,
                   shape,
                   hidden_act='ReLU',
                   loss=None,
                   sigmas=None,
                   *args, **kwargs) -> 'mlp_regressor':
        kwargs.update({'out_act' : 'linear'})
        return super().from_shape(
            shape=shape,
            hidden_act=hidden_act,
            loss=loss,
            sigmas=sigmas,
            *args, **kwargs
        )

    @classmethod
    def fit(cls,
            X_train:np.ndarray,
            T_train:np.ndarray,
            hidden_shape: Sequence[int],
            hidden_act='ReLU',
            loss=None,
            sigmas=None,
            *args, **kwargs):
        return super().fit(
            X_train=X_train, T_train=T_train, hidden_shape=hidden_shape,
            hidden_act=hidden_act,
            loss=loss, sigmas=sigmas, *args, **kwargs
        )
            
        
            
        
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

    def __repr__(self):
        return (f"<{self.__class__.__name__} optimizer with "
                f"eta0={self.eta0}, eta={self.eta}, lamb={self.lamb}>")

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

    def __repr__(self):
        return super().__repr__().replace(">", f", momentum={self.momentum}>")
        
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
    """An activation function class.
    """
    
    param : float = 1.0
    # 出力層の活性化関数として用いた場合の対応する損失関数クラス
    loss_type : Type['loss_func'] = dataclasses.field(init=False, default=None, repr=False)
    # 正準連結関数かどうか
    is_canonical : bool = dataclasses.field(init=False, default=None)

    def __call__(self, u):
        """活性化関数の値h(u)そのもの."""
        raise NotImplementedError

    def val2deriv(self, z):
        """活性化関数の関数値z = h(u)の関数として導関数h'(u)の値を計算する."""
        raise NotImplementedError

    def copy(self):
        return copy.copy(self)

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
    """Linear (identity) function.
    """
    def __post_init__(self):
        self.loss_type = mean_square

    def __call__(self, u):
        return self.param * u
    
    def val2deriv(self, z):
        return self.param


@dataclasses.dataclass
class step(act_func):
    """Heaviside's step function.
    """
    def __post_init__(self):
        self.loss_type = mean_square # 便宜上
        
    def __call__(self, u):
        return np.maximum(np.sign(u), 0.0)
    
    def val2deriv(self, z):
        return 1.0 # for simple perceptron


@dataclasses.dataclass
class sigmoid(act_func):
    """Logistic sigmoid function.
    """
    def __post_init__(self):
        self.loss_type = cross_entropy
        
    def __call__(self, u):
        return 0.5 * (1.0 + np.tanh(0.5 * self.param * u))
    
    def val2deriv(self, z):
        return self.param * z * (1.0 - z)


@dataclasses.dataclass
class tanh(act_func):
    """Hyperbolic tangent function.
    """
    def __call__(self, u):
        return np.tanh(self.param * u)
    
    def val2deriv(self, z):
        return self.param * (1.0 - z*z)


    
@dataclasses.dataclass
class ReLU(act_func):
    """ReLU: Rectified Linear Unit.
    """
    def __call__(self, u):
        return np.maximum(self.param * u, 0.0)
    
    def val2deriv(self, z):
        return np.where(z > 0.0, self.param, 0.0)


@dataclasses.dataclass
class LeakyReLU(act_func):
    """Leaky ReLU activation.
    """
    alpha:float = dataclasses.field(default=0.01)
        
    def __call__(self, u):
        return np.maximum(self.param * u, self.alpha * u)
    
    def val2deriv(self, z):
        return np.where(z > 0, self.param, self.alpha)


@dataclasses.dataclass
class softmax(act_func):
    """Softmax function.
    """
    def __post_init__(self):
        self.loss_type = mul_cross_entropy
        
    def __call__(self, u):
        tmp = u - u.max()
        return np.exp(tmp) / np.sum(np.exp(tmp), axis=1, keepdims=True)
    
    def val2deriv(self, z):
        return np.array([np.diag(z[i]) - np.outer(z[i], z[i]) for i in range(len(z))])


ACTIVATIONS = {
    'identity'   : linear,
    'linear'     : linear,
    'step'       : step,
    'threshold'  : step,
    'sigmoid'    : sigmoid,
    'logistic'   : sigmoid,
    'tanh'       : tanh,
    'relu'       : ReLU,
    'ReLU'       : ReLU,
    'LeakyReLU'  : LeakyReLU,
    'leakyrelu'  : LeakyReLU,
    'softmax'    : softmax
}


    
@dataclasses.dataclass
class loss_func:
    """Loss function of a neural network.
    """
    
    net: mlp = dataclasses.field(init=False, default=None, repr=False)
    
    """損失関数"""    
    def __call__(self, x, t):
        """
        損失関数の値.

        Parameter(s)
        ------------
        x : 教師データの入力
        t : 教師データの出力
        """
        if x is not None:
            self.net.forward_prop(x)
        return self._call_impl(t)

    def _call_impl(self, t):
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
    """Mean-Square Error.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        SSE = 0.5 * np.linalg.norm(self.net[-1].z - t)**2
        return SSE / batch_size

    
class cross_entropy(loss_func):
    """Cross entropy error for binary classification.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        z = self.net[-1].z
        return -(t * np.log(z) + (1.0 - t) * np.log(1 - z)).sum() / batch_size

    
class mul_cross_entropy(cross_entropy):
    """Cross entropy error for multiclass classification.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        return -np.sum((t * np.log(self.net[-1].z + 1e-7))) / batch_size


LOSSES = {
    'mean_square'       : mean_square,
    'cross_entropy'     : cross_entropy,
    'mul_cross_entropy' : mul_cross_entropy,
    None                : lambda: None
}




@dataclasses.dataclass
class logger:
    """学習経過の記録と損失のグラフの描画, および早期終了の制御
    """
    
    net : mlp                  = dataclasses.field(default=None, repr=False)
    loss: Sequence[float]      = dataclasses.field(default_factory=list, repr=False)
    n_sample : int             = None
    batch_size: int            = None
    X_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    T_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    X_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    T_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    _validate : bool   = dataclasses.field(default=False, repr=False)
    val_loss: Sequence[float]  = dataclasses.field(default_factory=list, repr=False)
    color : str                = dataclasses.field(default='tab:blue', repr=False)
    color2 : str               = dataclasses.field(default='tab:orange', repr=False)
    how: str                   = dataclasses.field(default='plot', repr=False)
    delta_epoch : int          = dataclasses.field(default=10, repr=False)
    early_stopping: bool       = dataclasses.field(default=True)
    patience_epoch: int        = 10
    tol: float                 = 1e-5
    best_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    stop_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    AIC : float                = dataclasses.field(init=False, default=None)
    BIC : float                = dataclasses.field(init=False, default=None)
    time: float                = None
    optimizer: _optimizer_base = None
    callback : Callable        = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        self.accumulated_loss = 0
        
        # whether to compute the values of loss for validation set (X_val, T_val)
        if not((self.X_val is None) and (self.T_val is None)):
            self._validate = True
            self.val_accuracy = []
                
        self.iterations = 0  # iterations so far
        self.epoch      = -1 # epochs so far
        # numbers of iterations per epoch
        self._iter_per_epoch = int(np.ceil(self.n_sample / self.batch_size))
        
        # how to show the values of loss function
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

        # Graph of loss
        if self._plot:
            self.fig, self.ax, self.secax = self.init_plot()
            plt.ion()

        # Early Stopping
        self.best_loss = np.inf
        if self._validate:
            self.best_val_loss = np.inf

        # start time of training
        self._t0 = time.time()
    
    def init_plot(self):
        fig, ax = plt.subplots(constrained_layout=True)        
        ax.set(xlabel='epochs', ylabel='loss')
        if self.iterations == 0:
            ax.set_xlim(0, self.delta_epoch)
            ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--')
        if self._validate and hasattr(self.net, 'test'):
            secax = ax.secondary_yaxis('right', functions=(self._to_percent, self._from_percent))
            secax.set_ylabel('accuracy [%]')
        else:
            secax = None
        
        return fig, ax, secax

    def _plot_every_epoch(self, logstr):            
        if self.epoch % self.delta_epoch == 1:
            self.ax.set_xlim(0, self.epoch + self.delta_epoch - 1)
            if self._validate:
                max_loss = max(self.loss + self.val_loss)
                if hasattr(self.net, 'test'):
                    max_loss = 0
            else:
                max_loss = max(self.loss)
            self.ax.set_ylim(0, max(max_loss, 1))

        x = [self.epoch - 1, self.epoch]
            
        if self._validate:
            self.ax.plot(x, self.val_loss[-2:], c=self.color2, label='validation loss')
            if hasattr(self.net, 'test'):
                self.ax.plot(self.val_accuracy, c=self.color2, linestyle='--', label='validation accuracy')
                
        self.ax.plot(x, self.loss[-2:], c=self.color, label='training loss')
        self.ax.set_title(logstr, fontsize=8)

        plt.show()
        plt.pause(0.2)

    def _plot_legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0]
        if self._validate:
            order = [1, 0]
            if hasattr(self, 'val_accuracy'):
                if self.val_accuracy:
                    order = [2, 0, 1]
                    
        plt.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order],
            bbox_to_anchor=(1, 0.5), loc='upper right', borderaxespad=1
        )
    
    def __call__(self):
        T_mini = self.net[-1].z - self.net[-1].delta
        self.accumulated_loss += self.net.loss(None, T_mini)
        
        # at the top of epoch
        if ((self.iterations >= self._iter_per_epoch)
            and (self.iterations % self._iter_per_epoch == 0)):
            
            self.epoch += 1
            self.loss.append(self.accumulated_loss / self._iter_per_epoch)
            self.accumulated_loss = 0

            # loss for the validation set (X_val, T_val)
            if self._validate:
                if self.net.dropout:
                    # if dropout is on, turn it off temporarily
                    self.net._set_training_flag(False)
                    
                self.val_loss.append(self.net.loss(self.X_val, self.T_val))
                if hasattr(self.net, 'test'):
                    accuracy = self.net.test(self.X_val, self.T_val, False)
                    self.val_accuracy.append(accuracy)
                
                if self.net.dropout:
                    self.net._set_training_flag(True)

            # log output
            logstr = f"Epoch {self.epoch}: Loss={self.loss[-1]:.3e}"
            if self._validate:
                logstr += f" (training), {self.val_loss[-1]:.3e} (validation)"
                if self.val_accuracy:
                    logstr += f", Accuracy={self.val_accuracy[-1]*100:.2f}% (validation)"
            
            if self._stdout:
                print(logstr)

            if self._plot:
                if self.epoch >= 1:    
                    self._plot_every_epoch(logstr)
                if self.epoch == 1:
                    self._plot_legend()

            # Early Stopping
            last_loss = self.loss[-1]            
            if self._validate:
                last_val_loss = self.val_loss[-1]
                # 検証用データに対する損失に一定以上の改善が見られない場合
                if last_val_loss > (self.best_val_loss - self.tol):
                    self._no_improvement_epoch += 1
                else:
                    self._no_improvement_epoch = 0
                # 現時点までの暫定最適値を更新(検証用データ) 
                if last_val_loss < self.best_val_loss:
                    self.best_val_loss = last_val_loss
                    self.best_params_val = self.net.get_params()
                    
            else:
                # 訓練データに対する損失に一定以上の改善が見られない場合
                if last_loss > (self.best_loss - self.tol):
                    self._no_improvement_epoch += 1
                else:
                    self._no_improvement_epoch = 0
            
            # 現時点までの暫定最適値を更新(訓練データ)
            if last_loss < self.best_loss:
                self.best_loss = last_loss
                self.best_params = self.net.get_params()

            if self._no_improvement_epoch > self.patience_epoch:
                which = 'Validation' if self._validate else 'Training'
                no_improvement_msg = (
                    f"{which} loss did not improve more than "
                    f"tol={self.tol} for the last {self.patience_epoch} epochs"
                    f" ({self.epoch} epochs so far)."
                )
                self.stop_params = self.net.get_params()
                if self.early_stopping:
                    raise NoImprovement(no_improvement_msg)

                warnings.warn(no_improvement_msg)

            # callback
            if self.callback is not None:
                self.callback(self.net)

        self.iterations += 1
        
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
    
    def plot(self, color='tab:blue', color2='tab:orange', *args, **kwargs):
        """学習のあと、グラフをふたたび表示
        """
        fig, ax, secax = self.init_plot()
        
        if self.val_loss:
            ax.plot(self.val_loss, color=color2, label='valid.loss', *args, **kwargs)
            ax.plot(self.val_accuracy, c=self.color2, linestyle='--', label='valid.accuracy')

        ax.plot(self.loss, color=color, label='train.loss', *args, **kwargs)
        ax.set_ylim(bottom=0)
        
        self._plot_legend()
        return fig, ax, secax

    # __________________________ for pickle _________________________
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('fig', None)
        state.pop('ax', None)
        state.pop('secax', None)
        state.pop('callback', None)
        return state

    def save(self, filename):
        utils.save(self, filename)


    # ________________________ for init_plot() _______________________
    @classmethod
    def _to_percent(cls, x):
        return 100 * x
    
    @classmethod
    def _from_percent(cls, x):
        return 0.01 * x

