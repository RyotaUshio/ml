"""Classification Algorithms.
"""

import numpy as np
from scipy.stats import mode
import scipy.linalg
from typing import Sequence
import dataclasses
import warnings

from . import utils, base, nn
from .nn import mlp_classifier


@dataclasses.dataclass(repr=False)
class k_nearest(base._estimator_base, base.classifier_mixin):
    """Classification with the k-nearest neighbors algorithm.
    """    
    X_train     : np.ndarray
    T_train     : np.ndarray
    k           : int = 1

    def __post_init__(self):
        if not (self.k >= 1 and isinstance(self.k, int)):
            raise ValueError("'k' must be a positive integer.")

    def __repr__(self):
        return (f"<{self.__class__.__name__} (k={self.k})"
                f" with {len(self.X_train)} pieces of training data>")

    def __call__(self, X : np.ndarray):
        X = utils.check_twodim(X)
        k_nearest_list = []
        for x in X:
            k_nearest = self.partial_call(x)
            k_nearest_list.append(k_nearest)

        return self.agg(k_nearest_list)
            
    def partial_call(self, x : np.ndarray):
        dist = scipy.linalg.norm(self.X_train - x, axis=1)
        idx = np.argsort(dist)[:self.k]
        k_nearest = self.T_train[idx]
        return k_nearest

    def agg(self, k_nearest_list):
        res = np.array(k_nearest_list)
        
        if utils.is_one_of_K(self.T_train):
            return np.argmax(np.sum(res, axis=1), axis=1)

        else:
            return mode(res, axis=1).mode.ravel()

    def predict_label(self, X):
        return self.__call__(X)

        


@dataclasses.dataclass(repr=False)
class generative(base._estimator_base, base.classifier_mixin):
    """Classification with generative models.
    """
    X_train     : dataclasses.InitVar[np.ndarray]
    T_train     : dataclasses.InitVar[np.ndarray]
    means       : np.ndarray = dataclasses.field(init=False)
    covs        : np.ndarray = dataclasses.field(init=False)
    priors      : np.ndarray = dataclasses.field(init=False)
    calc_mean   : dataclasses.InitVar[bool] = True
    calc_cov    : dataclasses.InitVar[bool] = True
    calc_prior  : dataclasses.InitVar[bool] = True
    dim         : int = dataclasses.field(init=False)
    n_class     : int = dataclasses.field(init=False)
    n_sample    : int = dataclasses.field(init=False)
    classification_type: str = 'multi'

    def __post_init__(self, X_train, T_train, calc_mean, calc_cov, calc_prior):
        self.n_sample = len(X_train)

        self.means, self.covs, self.priors = (
            utils.estimate_params(
                X_train, T_train,
                mean=calc_mean, cov=calc_cov, prior=calc_prior
            )
        )
        self.n_class, self.dim = self.means.shape
        
    def __call__(
            self,
            x : Sequence,
            cov_equal : bool=False,
            cov_identity : bool=False,
            prior_equal : bool=False
    ):
        """Computes the values of the discriminant function of each class.

        Parameters
        ----------
        x : array_like of shape (n_sample, n_feature)
            The input data matrix. Each row is supposed to represent a feature vector.
        cov_equal : bool, default=False
            Whether to assume that all classes have the same covariance matrix. 
            If True, linear discriminant function will be used, and quadratic otherwise.
            This parameter is set to True if `cov_identity == True`.
        cov_identity : bool, default=False
            Whether to assume that covariance matrices of all classes are identity
        prior_equal : bool, default=False
            Whether to assume that the prior probabilities of all classes are equal.

        Returns
        -------
        np.ndarray of shape (n_sample, n_classes)
            An array that contains the values of the discriminant function of each class.
        """
        discriminants = np.empty((len(x), self.n_class))

        for i in range(self.n_class):
            discriminants[:, i] = self.ith_discriminant(
                i, x,
                cov_equal=cov_equal,
                cov_identity=cov_identity,
                prior_equal=prior_equal
            )

        return discriminants

    def ith_discriminant(self, i, x, cov_equal, cov_identity, prior_equal):
        """Discriminant function of class i.
        """
        if cov_identity:
            cov_equal = True
            
        if cov_equal:
            return self.ith_linear(i, x, cov_identity, prior_equal)
        else:
            return self.ith_quadratic(i, x, prior_equal)
            
    def ith_linear(self, i, x, cov_identity, prior_equal):
        """Linear discriminant function of class i.
        """
        x = np.array(x).reshape((-1, self.dim))
        
        mu = self.means[i]
        prior = self.priors[i]

        if cov_identity:
            discriminant = (x - 0.5 * mu) @ mu.T
        else:
            sigma0 = np.mean(self.covs, axis=0)
            sigma_inv = scipy.linalg.inv(sigma0)
            discriminant = (x - 0.5 * mu) @ sigma_inv @ mu.T
            
        if not prior_equal:
            discriminant += np.log(prior)

        return discriminant

    def ith_quadratic(self, i, x, prior_equal):
        """Quadratic discriminant function of class i.
        """
        x = np.array(x).reshape((-1, self.dim))
        
        mu = self.means[i]
        sigma_inv = scipy.linalg.inv(self.covs[i])
        sigma_det = scipy.linalg.det(self.covs[i])
        prior = self.priors[i]

        mahalanobis_sq = np.diag((x - mu) @ sigma_inv @ (x - mu).T)
        discriminant = -0.5 * mahalanobis_sq - 0.5 * np.log(sigma_det)
        
        if not prior_equal:
            discriminant += np.log(prior)
            
        return discriminant

            
                             
        
@dataclasses.dataclass
class simple_perceptron(nn.mlp, base.classifier_mixin):
    classification_type: str = dataclasses.field(init=False)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.classification_type = 'binary'
        if self[-1].size > 1:
            warnings.warn(
                "Simple perceptron is an algorithm for binary classification, "
                "not multiclass classification."
            )
            self.classification_type = 'multi'
            
    @classmethod
    def fit(cls,
            X_train : np.ndarray,
            T_train : np.ndarray,
            sigmas=None,
            **kwargs):

        for argname in ['act_funcs', 'hidden_act', 'out_act', 'loss', 'hidden_shape']:
            if argname in kwargs:
                raise Exception(
                    f"Parameter '{argname}' is unavailable in '{self.__class__.__name__}'."
                )

        return super().fit(
            X_train=X_train, T_train=T_train,
            hidden_shape=[],
            act_funcs=[None, 'threshold'],
            loss='mean_square',
            **kwargs
        )

    def predict_label(self, x):
        return super().predict_label(x, threshold=0.5)
