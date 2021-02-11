import numpy as np
from typing import Type, Sequence, List, Callable
import warnings

from .. import base, utils



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
            from ._nn import act_func
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
