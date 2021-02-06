"""Evaluation of Estimators.
"""

import numpy as np
import scipy.linalg
import pandas as pd
from typing import Type, Sequence, List, Callable
import warnings

from . import base, utils, feature
from .exceptions import EmptyCluster


def evaluate(estimator: Type[base._estimator_base], *args, type=None, **kwargs):
    if type is None:
        type = estimator.estimator_type
    return EVALUATORS[type](estimator, *args, **kwargs)



class _evaluator_base:
    def __init__(self, estimator, *args, **kwargs):
        self.estimator = estimator
        self.eval(*args, **kwargs)
        for k, v in self.measures.items():
            setattr(self, k, v)

    def __repr__(self):
        ret = f"<{self.estimator.__class__.__name__} evaluator>\n"
        for k, v in self.measures.items():
            if isinstance(v, pd.Series):
                ret += f"{k} : \n{pd.DataFrame(v, columns=['']).T}\n"
            else:
                ret += f"{k} : {v:.6f}\n"
        return ret.strip('\n')

    def __getitem__(self, key):
        return self.measures.__getitem__(key)

        
        
        
class classifier_evaluator(_evaluator_base):
    def __repr__(self):
        if self.kwargs:
            repr_kwargs = 'kwargs : '
            for k, v in self.kwargs.items():
                repr_kwargs += f"{k}={v}, "
            repr_kwargs = repr_kwargs[:-2]
            return super().__repr__().replace('>\n', '>\n' + repr_kwargs + '\n')
        return super().__repr__()
    
    def eval(self, X, T, margins=True, **kwargs):
        self.conf = self.make_confusion_matrix(X, T, margins, **kwargs)
        self.measures = self.agg(self.conf)
        self.kwargs = kwargs

    def make_confusion_matrix(self, X, T, margins=True, **kwargs):
        true = utils.digit(T)
        if hasattr(self.estimator, 'predict_label'):
            predicted = self.estimator.predict_label(X, **kwargs)
        elif hasattr(self.estimator, '__call__'):
            predicted = self.estimator(X, **kwargs)

        if not utils.is_digit(predicted):
            raise Exception(
                "Make sure `estimator.predict_label()` or `estimator.__call__` "
                "has been implemented and they return predicted class labels in digit."
            )

        df = pd.DataFrame({'predicted': predicted.ravel(), 'true': true.ravel()})
        conf = pd.crosstab(df['predicted'], df['true'], margins=margins)
        return conf

    @staticmethod
    def agg(conf):
        if 'All' in conf.index:
            conf = conf.drop(index='All')
        if 'All' in conf.columns:
            conf = conf.drop(columns='All')
            
        TP = np.diag(conf.values)
        precision = TP / conf.sum(axis=1)
        recall    = TP / conf.sum(axis=0)
        f1 = 2 * precision * recall / (precision + recall)
        
        measures = dict(precision=precision, recall=recall, f1=f1)
        for k in measures:
            measures[k].index.name = 'class'

        n_class = len(conf.index)
        if n_class <= 2:
            return measures

        for name in ['precision', 'recall', 'f1']:
            measures[f'macro_{name}'] = measures[name].mean()
            measures[f'weighted_average_{name}'] = np.average(
                measures[name], weights=conf.sum(axis=0)
            )
        measures[f'micro_average'] = TP.sum() / conf.values.sum()

        return measures

        


class cluster_evaluator(_evaluator_base):
    def eval(self, silhouette=True, CH=True, DB=True, **kwargs):
        try:
            self.estimator.check_empty_cluster()
        except EmptyCluster as e:
            print(e)
            self.n_cluster = len(np.unique(self.estimator.labels))
        else:
            self.n_cluster = self.estimator.k
        
        self.measures = dict()
        if silhouette:
            self.measures['silhouette'] = self.eval_silhouette(**kwargs)
        if CH:
            self.measures['CH'] = self.eval_Calinski_Harabasz()
        if DB:
            self.measures['DB'] = self.eval_Davies_Boundin(**kwargs)
        

    def eval_silhouette(self, **kwargs):
        """A Higher value means that the clustering is better.
        """
        X = self.estimator.X
        labels = utils.digit(self.estimator.labels)
        results = []

        for x, label in zip(X, labels):
            
            b_min = np.inf
            
            for i, class_i in enumerate(utils.class_iter(X, labels)):

                if i == label:
                    # x belongs to class_i
                    a = np.sum(scipy.linalg.norm(class_i - x, axis=1, **kwargs))
                    a /= (len(class_i) - 1)

                else:
                    # x does not belong to class_i
                    b = np.mean(scipy.linalg.norm(class_i - x, axis=1, **kwargs))
                    if b < b_min:
                        b_min = b

            numer = b_min - a
            denom = max(a, b_min)
            results.append(numer / denom)

        return np.array(results)

    def eval_Calinski_Harabasz(self):
        """A Higher value means that the clustering is better.
        """
        X = self.estimator.X
        labels = self.estimator.labels
        k = self.n_cluster
        n = len(X)
        cov_w, cov_b = feature.lda.within_between_cov(X, labels)
        CH = np.trace(cov_b) / np.trace(cov_w) * (n - k) / (k - 1)
        return CH

    def eval_Davies_Boundin(self, **kwargs):
        """A lower value means that the clustering is better.
        """
        X = self.estimator.X
        labels = self.estimator.labels
        centroids = self.estimator.centroids
        k = self.n_cluster
        s = np.zeros(k)
        d = np.zeros((k, k))

        for i, class_i in enumerate(utils.class_iter(X, labels)):
            s[i] = np.mean(scipy.linalg.norm(class_i - centroids[i], axis=1, **kwargs))
            
            for j in range(k):
                d[i, j] = scipy.linalg.norm(centroids[i] - centroids[j], **kwargs)

        similarity = []
        for i in range(k):
            val = []
            for j in range(k):
                if i != j:
                    val.append((s[i] + s[j]) / d[i, j])
            similarity.append(max(val))

        DB = np.mean(similarity)
        return DB
        
class regressor_evaluator(_evaluator_base):
    def eval(self, X, T):
        n_sample = len(X)
        true = T
        predicted = self.estimator(X)
        RMS = scipy.linalg.norm(predicted - true) / np.sqrt(n_sample)
        SSE = 0.5 * RMS**2 * n_sample

        self.SSE = SSE
        self.RMS = RMS
    




EVALUATORS = {
    'classifier' : classifier_evaluator,
    'cluster'    : cluster_evaluator,
    'regressor'  : regressor_evaluator
}
