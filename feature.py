"""Feature Extraction Processors."""

import numpy as np
import dataclasses
from typing import Type, Sequence, List, Callable

import nn
import utils



class transformer:
    def __init__(self, X):
        self.X = X

    def transform(self):
        pass


class linear_transformer(transformer):
    def __init__(self, X, basis=None, ctrb=None):
        super().__init__(X)
        self.basis = basis
        self.ctrb = ctrb

    def transform(self):
        if self.basis is not None:
            self.X_trans = project(self.X, self.basis)


class PCA(linear_transformer):
    def __init__(self, X):
        basis, ctrb = pca(X)
        super().__init__(X, basis, ctrb)

        
class LDA(linear_transformer):
    def __init__(self, X):
        basis, ctrb = lda(X)
        super().__init__(X, basis, ctrb)
        


class _autoencoder_base(nn.mlp):
    def train(self, X_train:np.ndarray, **kwargs) -> None:
        kwargs.update({'T_train' : X_train})
        if 'X_val' in kwargs:
            kwargs.update({'T_val' : kwargs['X_val']})
        super().train(X_train=X_train, **kwargs)

    @classmethod
    def fit(cls,
            X_train:np.ndarray,
            hidden_shape_half: Sequence[int],
            encode_act='ReLU',
            decode_act='sigmoid',
            **kwargs):
        if 'X_val' in kwargs:
            kwargs.update({'T_val' : kwargs['X_val']})

        for argname in ['act_funcs', 'hidden_act', 'out_act']:
            if argname in kwargs:
                raise Exception(
                    f"Parameter '{argname}' is unavailable in autoencoders."
                    " Use 'encode_act' & 'decode_act' instead."
                )

        encode_act, decode_act = cls._make_activation_list(
            hidden_shape_half=hidden_shape_half, encode_act=encode_act, decode_act=decode_act
        )
        
        return super().fit(
            X_train=X_train, T_train=X_train,
            hidden_shape=hidden_shape_half + hidden_shape_half[-2::-1],
            act_funcs=[None] + encode_act + decode_act[::-1],
            loss='mean_square',
            **kwargs
        )

    @staticmethod
    def _make_activation_list(hidden_shape_half, encode_act, decode_act):
        if isinstance(encode_act, (str, nn.act_func)):
            encode_act = [encode_act for _ in range(len(hidden_shape_half))]
        encode_act = list(encode_act)
        
        if isinstance(decode_act, (str, nn.act_func)):
            decode_act = [decode_act for _ in range(len(hidden_shape_half))]
        decode_act = list(decode_act)

        return encode_act, decode_act


@dataclasses.dataclass
class autoencoder(_autoencoder_base):
    """Autoencoder.

    Examples
    --------
    >>> (X_train, T_train), (X_test, T_test) = utils.load_data()
    >>> ae = ft.autoencoder.fit(X_train[10000:], [128, 64, 32], encode_act='ReLU', decode_act=['sigmoid', 'ReLU', 'ReLU'], X_val=X_train[:10000])
    ...
    >>> output = ae(X_train)
    >>> utils.imshow(output[0])
    """
    simple_aes : Sequence[_autoencoder_base] = dataclasses.field(default_factory=list, repr=False)
    
    @classmethod
    def fit(cls,
            X_train,
            hidden_shape_half,
            encode_act='ReLU',
            decode_act='sigmoid',
            verbose=True,
            layer_by_layer=True,
            **kwargs):

        if layer_by_layer:
            encode_act, decode_act = cls._make_activation_list(
                hidden_shape_half=hidden_shape_half, encode_act=encode_act, decode_act=decode_act
            )
            nets = []
            for i, (n_hidden_unit, encode_act, decode_act) in enumerate(zip(hidden_shape_half, encode_act, decode_act)):
                if verbose:
                    print(f"Training simple autoencoder {i+1}/{len(hidden_shape_half)}...")

                if i >= 1:
                    nets[-1](X_train)
                    X_train = nets[-1][1].z
                    if 'X_val' in kwargs:
                        nets[-1](kwargs['X_val'])
                        kwargs['X_val'] = nets[-1][1].z

                nets.append(
                    super().fit(
                        X_train,
                        hidden_shape_half=[n_hidden_unit],
                        encode_act=encode_act,
                        decode_act=decode_act,
                        **kwargs
                    )
                )

            if verbose:
                print("Training has finished.")
                
            simple_aes = [net.copy() for net in nets]
            layers = []
            while nets:
                net = nets.pop()
                layers = [net[1]] + layers + [net[2]]
            layers = [net[0]] + layers

            return cls(layers=layers, loss=net.loss, simple_aes=simple_aes)

        return super().fit(
            X_train=X_train,
            hidden_shape_half=hidden_shape_half,
            encode_act=encode_act,
            decode_act=decode_act,
            **kwargs
        )

            
  

    

def project(X, basis):
    r"""与えられたデータ集合を、与えられた基底に射影して変換する。

    Parameters
    ----------
    X : array_like
        データを行として並べた行列
    basis : array_like
        基底を行として並べた行列

    Returns
    -------
    np.ndarray
        射影後のデータを行として並べた行列
    """
    X_projected = np.matmul(X, basis.T)
    if np.all(np.imag(X_projected) == 0):
        return X_projected.astype(float)
    return X_projected



def pca(X):
    r"""主成分分析を行う.

    Parameters
    ----------
    X : array_like
        データを行として並べた行列

    Returns
    -------
    eigvecs : np.ndarray
        分散共分散行列の固有ベクトルを行として並べた行列. 寄与率の大きい順にソートされている.
    ctrb : np.ndarray
        各固有ベクトルの寄与率を、eigvecsに対応した順番で並べた配列."""
    
    sigma = np.cov(X, rowvar=False) # 分散共分散行列
    eigvals, eigvecs = np.linalg.eig(sigma) # 固有値・固有ベクトル
    # np.linalg.eigは固有ベクトルを列とするarrayを返すので、転置をとる.
    eigvecs = eigvecs.T
    
    # 固有値の大きい順に固有値と固有ベクトルをソートする
    idx = np.flip(np.argsort(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx]
    
    ctrb = eigvals / eigvals.sum()  # 各固有ベクトルの寄与率
    
    return eigvecs, ctrb


    
