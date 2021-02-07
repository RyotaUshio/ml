"""Clustering Algorithms.
"""

import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
from typing import Sequence, List, Type
import dataclasses
import warnings

from . import utils, base, nn, classify
from .exceptions import EmptyCluster, NoImprovement



@dataclasses.dataclass(repr=False)
class k_means(base._estimator_base, base.cluster_mixin):
    X             : np.ndarray
    k             : int
    plot          : dataclasses.InitVar[bool] = True
    delta         : dataclasses.InitVar[float] = 0.7
    verbose       : dataclasses.InitVar[bool] = True
    n_sample      : int = dataclasses.field(init=False)
    centroids     : np.ndarray = dataclasses.field(init=False)
    labels        : np.ndarray = dataclasses.field(init=False)

    def __post_init__(self, plot, delta, verbose):
        self.n_sample = len(self.X)
        self.__call__(plot=plot, delta=delta, verbose=verbose)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} (k={self.k}) with {self.n_sample} pieces of data>"
 
    def __call__(
            self,
            plot=True,
            tol: float=1e-2,
            patience_iter: int=5,
            delta: float=0.7,
            verbose: bool=True
    ) -> None:
        
        self.set_initial(delta)
        centroid_labels = np.array(range(self.k))
        self.labels = np.zeros(self.n_sample)
        count = 1

        if plot:
            fig = plt.figure()
            plt.ion()

        while True:
            try:
                if verbose:
                    print('\r' + f'...Loop {count}...', end='')

                # (can be interpreted as) the E step of the EM algorithm
                knn = classify.k_nearest(self.centroids, centroid_labels, k=1)
                new_labels = knn(self.X)

                # break if the assignments do not change
                if np.sum(np.abs(new_labels - self.labels)) == 0:
                    break
                self.labels = new_labels

                self.check_empty_cluster()

                if plot:
                    plt.gca().remove()
                    utils.scatter(self.X, self.labels, fig=fig)
                    plt.pause(0.8)

                # (can be interpreted as) the M step of the EM algorithm
                means, _, _ = utils.estimate_params(self.X, self.labels, cov=False, prior=False)
                self.centroids = means
                count += 1

            except EmptyCluster as e:
                self.set_initial(delta)
                warnings.warn(f'{e} The centroids were re-initialized.')
                continue
                
            except KeyboardInterrupt:
                warnings.warn("Clustering stopped by user.")
                break

        if verbose:
            print('\r' + f'Finished after {count} loops.')

            

@dataclasses.dataclass(repr=False)
class em(base._estimator_base, base.cluster_mixin):
    X             : np.ndarray
    k             : int
    # used in convergence test
    tol           : dataclasses.InitVar[float] = 0.1
    patience_iter : dataclasses.InitVar[int] = 5
    no_improvement_iter : int = dataclasses.field(init=False, default=0)
    # used in initilization with K-means
    delta         : dataclasses.InitVar[float] = 0.7
    # whether to print current number of iterations
    verbose       : dataclasses.InitVar[bool] = True
    # Responsibilities : estimated in the E step
    resps         : np.ndarray = dataclasses.field(init=False)
    # Parameters of Gaussian mixture : estimated in the M step
    means         : np.ndarray = dataclasses.field(init=False)
    covs          : np.ndarray = dataclasses.field(init=False)
    priors        : np.ndarray = dataclasses.field(init=False)
    # result
    labels        : np.ndarray = dataclasses.field(init=False)
    centroids     : np.ndarray = dataclasses.field(init=False)
    n_sample      : int = dataclasses.field(init=False)
    
    def __post_init__(self, tol, patience_iter, delta, verbose):
        self.n_sample = len(self.X)
        self.__call__(tol=tol, patience_iter=patience_iter, delta=delta, verbose=verbose)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} (k={self.k}) with {self.n_sample} pieces of data>"

    def set_initial(self, delta):
        kmeans = k_means(X=self.X, k=self.k, plot=False, delta=delta, verbose=False)
        self.means, self.covs, self.priors = utils.estimate_params(X=self.X, T=kmeans.labels)
        self.resps = np.zeros((self.n_sample, self.k))
        self.joints = np.zeros((self.n_sample, self.k))
        self.log_likelihood = -np.inf

    def kth_gaussian(self, x, k):
        return scipy.stats.multivariate_normal.pdf(
            x,
            mean=self.means[k],
            cov=self.covs[k]
        )

    def E_step(self):
        for n, x in enumerate(self.X):
            gaussians = np.array([self.kth_gaussian(x, k) for k in range(self.k)])
            nth_joints = self.priors * gaussians
            self.joints[n] = nth_joints
            resp = nth_joints / nth_joints.sum()
            self.resps[n] = resp

    def M_step(self):
        for k in range(self.k):
            kth_resps = self.resps[:, k]
            self.means[k] = np.average(self.X, weights=kth_resps, axis=0)
            self.covs[k] = np.average(
                [np.outer(row, row) for row in self.X - self.means[k]],
                weights=kth_resps,
                axis=0
            )
        Nk = np.sum(self.resps, axis=0)
        self.prior = Nk / self.n_sample

    def update_log_likelihood(self):
        new_log_likelihood = np.sum(
            np.log(np.sum(self.joints, axis=1))
        )

        if new_log_likelihood < self.log_likelihood + self.tol:
            self.no_improvement_iter += 1
        else:
            self.no_improvement_iter = 0

        if self.no_improvement_iter > self.patience_iter:
            raise NoImprovement(
                f"Log likelihood did not improved more than {self.tol} "
                f"for the last {self.patience_iter} iterations."
            )
        
        self.log_likelihood = new_log_likelihood

    def __call__(
            self,
            tol: float=1e-2,
            patience_iter: int=5,
            delta: float=0.7,
            verbose: bool=True
    ) -> None:
        self.set_initial(delta)
        self.tol = tol
        self.patience_iter = patience_iter
        count = 0

        while True:
            count += 1
            if verbose:
                print('\r' + f'...Loop {count}...', end='')

            self.E_step()
            self.M_step()
            
            try:
                self.update_log_likelihood()
                if verbose:
                    print(f' Log likelihood = {self.log_likelihood}', end='')
                
            except NoImprovement as e:
                print(e)
                break
            
            except KeyboardInterrupt:
                warnings.warn("Clustering stopped by user.")
                break

        self.centroids = self.means
        self.labels = np.argmax(self.resps, axis=1)

        try:
            self.check_empty_cluster()
        except EmptyCluster as e:
            warnings.warn(f'{e}')

        if verbose:
            print('\r' + f'Finished after {count} loops.')


    
class mean_shift(base._estimator_base, base.cluster_mixin):
    pass


class competitive_layer(nn.layer):
    def __init__(self, W:Sequence=None):
        super().__init__(W=W, b=np.zeros_like(W[0]), h=competitive_activation())

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self.size} neurons>"

    
@dataclasses.dataclass
class competitive_activation(nn.act_func):
    def __post_init__(self):
        self.loss_type = competitive_loss

    def __call__(self, u):
        return np.where(u >= u.max(axis=1, keepdims=True), 1, 0)

    
class competitive_loss(nn.loss_func):
    def _call_impl(self, t):
        pass
    def error(self, t):
        pass

class competitive_logger(nn.logger):
    def __call__(self):
        self.iterations += 1

        if self.iterations % self._iter_per_epoch == 0:
            self.epoch += 1
            if self._stdout:
                print('\r' + f'...Epoch {self.epoch}...', end='')
        

class competitive_net(nn.mlp, base.cluster_mixin):
    def __init__(self, X: np.ndarray,  k: int, tol: float=0.03, delta=0.45, **kwargs):
        self.X = utils.check_twodim(X)
        self.k = k
        self.n_sample = len(self.X)
        self.X_normalized = self.X / scipy.linalg.norm(self.X, axis=1, keepdims=True)
        self.tol = tol
        self.set_initial(delta)
        
        input_layer = nn.layer(size=self.X.shape[1], first=True)
        W = self.centroids.T
        comp_layer = competitive_layer(W)
        layers = [input_layer, comp_layer]
        super().__init__(layers=layers)
        
        self.train(**kwargs)
        
        self.centroids = self[-1].W.T
        knn = classify.k_nearest(
            self.centroids, np.array(range(self.k)), k=1
        )
        self.labels = knn(self.X_normalized)
        try:
            self.check_empty_cluster()
        except EmptyCluster as e:
            print(e)

    def back_prop(self, t : np.ndarray) -> None:
        pass

    def set_gradient(self):
        self[-1].dJdW = -self[0].z.T @ self[-1].z
        self[-1].dJdb = 0
    
    def train(
            self, *,
            eta0:float=0.05,
            optimizer='SGD',
            max_epoch:int=100,
            verbose=True
    ) -> None:
        super().train(
            self.X_normalized, np.zeros_like(self.X[:, [0]]),
            eta0=eta0, optimizer=optimizer, max_epoch=max_epoch,
            batch_size=1, lamb=0, how='stdout' if verbose else 'off'
        )
        if verbose:
            print('\r' + f'Finished after {self.log.iterations // self.log._iter_per_epoch} epochs.')


    def train_one_step(self, X_mini, T_mini, optimizer):
        super().train_one_step(X_mini, T_mini, optimizer)
        if scipy.linalg.norm(optimizer.dWs[-1]) < self.tol:
            raise nn.NoImprovement(
                f"The centroids did not move more than {self.tol}."
            )
        self[-1].W /= scipy.linalg.norm(self[-1].W, axis=0, keepdims=True)

    def log_init(self, **kwargs):
        return competitive_logger(**kwargs)

    

@dataclasses.dataclass(repr=False)
class as_cluster(base.cluster_mixin):
    """Interpret given patterns and labels as a result of clutering, and
    returns an ojbect that ml.evaluate() can recognize as an cluster estimator.
    """
    X             : np.ndarray
    labels        : np.ndarray
    k             : int = dataclasses.field(init=False)
    n_sample      : int = dataclasses.field(init=False)
    centroids     : np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.n_sample = len(self.X)
        self.centroids, _, _ = utils.estimate_params(self.X, self.labels, cov=False, prior=False)
        self.k = len(self.centroids)
        
    def __repr__(self):
        return f"<{self.__class__.__name__} (k={self.k}) with {self.n_sample} pieces of data>"
