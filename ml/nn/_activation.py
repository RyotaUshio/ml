import numpy as np
import dataclasses
from typing import Type
import copy

from ._loss import loss_func, mean_square, cross_entropy, multi_cross_entropy


@dataclasses.dataclass
class act_func:
    """An activation function class.
    """
    
    param : float = 1.0
    # 出力層の活性化関数として用いた場合の対応する損失関数クラス
    loss_type : Type[loss_func] = dataclasses.field(init=False, default=None, repr=False)
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
        self.loss_type = multi_cross_entropy
        
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
