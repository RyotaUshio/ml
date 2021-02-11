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



class k_means(base._estimator_base, base.cluster_mixin):
    def __init__(
            self,
            X             : np.ndarray,
            k             : int,
            *,
            plot          : bool = True,
            delta         : float = 0.7,
            verbose       : bool = True
    ):
        self.X = X
        self.k = k
        self.n_sample = len(self.X)
        self.__call__(plot=plot, delta=delta, verbose=verbose)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} (k={self.k}) with {self.n_sample} pieces of data>"
 
    def __call__(
            self, *,
            delta: float=0.7,
            verbose: bool=True,
            plot=True
    ) -> None:
        
        self.set_initial(delta)
        centroid_labels = np.array(range(self.k))

        self.monitor = utils.stop_monitor(verbose=verbose, name='centroids')

        if plot:
            fig = plt.figure()
            plt.ion()

        while True:
            try:
                # (can be interpreted as) the E step of the EM algorithm
                knn = classify.k_nearest(self.centroids, centroid_labels, k=1)
                self.labels = knn(self.X)

                self.check_empty_cluster()

                if plot:
                    plt.cla()
                    self.scatter(fig=fig)
                    plt.pause(0.2)

                # (can be interpreted as) the M step of the EM algorithm
                means, _, _ = utils.estimate_params(self.X, self.labels, cov=False, prior=False)
                self.centroids = means
                self.monitor(self.centroids)

            except EmptyCluster as e:
                self.set_initial(delta)
                warnings.warn(f'{e} The centroids were re-initialized.')
                continue

            except NoImprovement as e:
                if verbose:
                    print(e)
                break
                
            except KeyboardInterrupt:
                warnings.warn("Clustering stopped by user.")
                break

        if plot:
            plt.ioff()
        if verbose:
            print('\r' + f'Finished after {self.monitor.count} loops.')

            


class em(base._estimator_base, base.cluster_mixin):
    def __init__(
            self, 
            X             : np.ndarray,
            k             : int,
            *,
            # used in convergence test
            least_improve : float = 1e-2,
            patience      : int = 5,
            # used in initilization with K-means
            delta         : float = 0.7,
            # whether to print current status
            verbose       : bool = True,
            # whether & how to plot current status
            plot          : bool = True,
            **kwargs
    ):
        self.X = X
        self.k = k
        self.n_sample = len(self.X)
        self.__call__(
            least_improve=least_improve,
            patience=patience,
            delta=delta,
            verbose=verbose,
            plot=plot,
            **kwargs
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__} (k={self.k}) with {self.n_sample} pieces of data>"

    def set_initial(self, delta):
        kmeans = k_means(X=self.X, k=self.k, plot=False, delta=delta, verbose=False)
        self.means, self.covs, self.priors = utils.estimate_params(X=self.X, T=kmeans.labels)
        self.resps = np.ones((self.n_sample, self.k))
        self.joints = np.zeros((self.n_sample, self.k))
        self.log_likelihood = -np.inf
        self.centroids = self.means

    def kth_gaussian(self, x, k):
        return scipy.stats.multivariate_normal.pdf(
            x,
            mean=self.means[k],
            cov=self.covs[k]
        )

    def E_step(self):
        gaussians = np.array([self.kth_gaussian(self.X, k) for k in range(self.k)]).T
        self.joints = self.priors * gaussians
        self.resps = self.joints / self.joints.sum(axis=1, keepdims=True)

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
        self.monitor(new_log_likelihood / (self.n_sample * self.k))

    def __call__(
            self, *,
            least_improve: float=1e-2,
            patience: int=5,
            delta: float=0.7,
            verbose: bool=True,
            plot: bool=True,
            **kwargs
    ) -> None:
        self.set_initial(delta)
        self.monitor = utils.improvement_monitor(
            least_improve=least_improve, patience=patience, better='higher', verbose=verbose, name='log likelihood'
        )

        if plot:
            fig = plt.figure()
            plt.ion()

        while True:
            try:
                if plot:
                    plt.cla()
                    self.scatter(fig=fig, step=1, **kwargs)
                    plt.pause(0.2)
                
                self.E_step()
                self.M_step()
                self.update_log_likelihood()
                
            except NoImprovement as e:
                if verbose:
                    print(e)
                break
            
            except KeyboardInterrupt:
                warnings.warn("Clustering stopped by user.")
                break

        self.labels = np.argmax(self.resps, axis=1)
        
        try:
            self.check_empty_cluster()
        except EmptyCluster as e:
            warnings.warn(f'{e}')

        if plot:
            plt.ioff()
        if verbose:
            print('\r' + f'Finished after {self.monitor.count} loops.')

    def scatter(self, how='soft', centroids=True, contours=False, step=0.1, **kwargs):
        if how not in ['soft', 'hard']:
            raise ValueError(f"'how' is expected to be either 'soft' or 'hard', got {how}")
        if how == 'hard':
            return super().scatter(centroids=centroids, **kwargs)
        colors = np.array(
            [np.average(utils.get_colors()[:self.k], weights=resp, axis=0) for resp in self.resps]
        ) 
        utils.scatter(self.X, c=colors, **kwargs)
        if centroids:
            self.scatter_centroids(ax=plt.gca())
        if contours:
            self.contour(ax=plt.gca(), step=step)

    def contour(self, step=0.1, sigmas=[1, 2], fig=None, ax=None):
        fig, ax = utils.make_subplot(fig, ax, False)
        levels = [sigma**2 for sigma in sigmas]
        for k in range(self.k):
            CS = utils.contour(
                lambda X: self.mahalanobis_sq(X, k=k),
                ax.get_xlim(), ax.get_ylim(), step, ax=ax, levels=levels,
                colors='w', linewidths=1
            )
            fmt = {}
            for l in CS.levels:
                fmt[l] = f'${np.sqrt(l):.0f}\\sigma$'
            ax.clabel(CS, fmt=fmt, fontsize=8)
        

            
    def mahalanobis_sq(self, X, k=None):
        if k is None:
            k = range(self.k)
        elif not hasattr(k, '__iter__'):
            k = [k]
        return np.array(
            [[(x - self.means[k]) @ scipy.linalg.inv(self.covs[k]) @ (x - self.means[k]).T for k in k]
             for x in X]
        )



    
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

        means, _, _ = utils.estimate_params(self.X, self.labels, cov=False, prior=False)
        self.centroids *= scipy.linalg.norm(means, axis=1, keepdims=True)

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
        # ____________ utils.change_monitorを使って書き直す_____________
        if scipy.linalg.norm(optimizer.dWs[-1]) < self.tol:
            raise nn.NoImprovement(
                f"The centroids did not move more than {self.tol}."
            )
        self[-1].W /= scipy.linalg.norm(self[-1].W, axis=0, keepdims=True)
        # ______________________________________________________________

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
