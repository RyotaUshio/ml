"""Feature Extraction & Dimensionality Reduction"""

import numpy as np
import scipy.linalg
import dataclasses
from typing import Sequence

from . import nn, utils




class transformer:
    def __init__(self, X):
        self.X = X

    def transform(self):
        pass

    def __getitem__(self, key):
        return self.X_trans[key]


    
class linear_transformer(transformer):
    def __init__(self, X, basis=None):
        super().__init__(X)
        self.basis = basis
        self.transform()

    def transform(self):
        if self.basis is None:
            raise Exception("`basis` is not yet set.")
        self.X_trans = np.matmul(self.X, self.basis.T)

        

class eigen_transformer(linear_transformer):
    def __init__(self, X, *args, **kwargs):
        eigvals, eigvecs = self.eigen(X, *args, **kwargs)
        
        # sort eigenvalues & eigenvectors in a descending order of eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[idx]
        
        self.eigvals = eigvals
            
        super().__init__(X, eigvecs)

    @staticmethod
    def eigen(X):
        """Solve some sort of eigenvalue problem.

        Returns
        -------
        eigvals, eigvecs
            `eigvecs` is an array whose rows represent eigenvectors.
        """
        raise NotImplementedError

    def reduce(self, dim : int):
        dim_max = self[:].shape[1]
        if dim > dim_max:
            raise ValueError(f"`dim` must be `dim < {dim_max}`.")
        return self[:, :dim]

        

class pca(eigen_transformer):
    def __init__(self, X):
        super().__init__(X)
        
        # contributon ratio
        self.ctrb = self.eigvals / np.sum(self.eigvals)
        
        # cumulative contribution ratio
        self.cumul_ctrb = self.ctrb.copy()
        for i in range(1, len(self.ctrb)):
            self.cumul_ctrb[i] += self.cumul_ctrb[i-1]
            
    @staticmethod
    def eigen(X):
        sigma = np.cov(X, rowvar=False)             # 分散共分散行列
        eigvals, eigvecs = scipy.linalg.eigh(sigma) # 固有値・固有ベクトル
        # np.linalg.eigは固有ベクトルを列とするarrayを返すので、転置をとる.
        return eigvals, eigvecs.T
        


class whiten(pca):
    def transform(self):
        super().transform()
        np.matmul(self.X_trans, np.diag(np.power(self.eigvals, -1/2)), out=self.X_trans)
    
        
class lda(eigen_transformer):
    def __init__(self, X, T):
        super().__init__(X, T)
        n_class = len(np.unique(T, axis=0))
        n_feature = X.shape[1]
        n_positive_eigval = min(n_class - 1, n_feature)
        self.X_trans = self.X_trans[:, :n_positive_eigval]
        
    @staticmethod
    def eigen(X, T):
        cov_within, cov_between = lda.within_between_cov(X, T)
        eigvals, eigvecs = scipy.linalg.eigh(a=cov_between, b=cov_within)
        return eigvals, eigvecs.T
        
    @staticmethod
    def within_between_cov(X, T):
        _, covs, ratio = utils.estimate_params(X, T, mean=False)
        cov_all = np.cov(X, rowvar=False)
        cov_within = np.average(covs, axis=0, weights=ratio)
        cov_between = cov_all - cov_within
        return cov_within, cov_between
        
        
        


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

    Attributes
    ----------
    simple_aes : list of autoencoders
        The simple autoencoders used in training. A simple autoencoder refers to an autoencoder 
        which has 1 hidden layer.

    Examples
    --------
    >>> (X_train, T_train), (X_test, T_test) = utils.load_data()
    >>> ae = ft.autoencoder.fit(
    ...     X_train[10000:], [128, 64, 32], 
    ...     encode_act='ReLU', decode_act=['sigmoid', 'ReLU', 'ReLU'], 
    ...     X_val=X_train[:10000]
    ... )
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

        if not layer_by_layer:
            return super().fit(
            X_train=X_train,
            hidden_shape_half=hidden_shape_half,
            encode_act=encode_act,
            decode_act=decode_act,
            **kwargs
        )
        
        else:
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
