"""Multi-layer Perceptron.
"""

import numpy as np
import dataclasses
from typing import Type, Sequence, List, Callable
import warnings
import pickle
import copy

from .. import base, utils
from ..exceptions import NoImprovement
from ._layers import layer, dropout_layer, inverted_dropout_layer
from ._activation import ACTIVATIONS, act_func, ReLU, LeakyReLU
from ._loss import LOSSES, loss_func
from ._optimizer import OPTIMIZERS
from ._logger import logger



    
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
            eta0:float=1e-3,
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
