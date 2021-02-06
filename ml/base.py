import numpy as np
import scipy.linalg

from . import utils
from .exceptions import EmptyCluster


class _estimator_base:
    pass



class classifier_mixin:
    estimator_type = 'classifier'

    def predict_label(self, x, threshold=None, **kwargs):
        """Return the predicted class labels, not inferior probability of each class.
        """
        output = self(x, **kwargs)
        if self.classification_type == 'multi':
            labels = utils.prob2label(output)
        elif self.classification_type == 'binary':
            labels = np.where(output >= threshold, 1, 0)
            
        return labels
    
    def predict_one_of_K(self, x):
        """Return the predicted class labels expressed in 1-of-K encoding.
        """
        return utils.one_of_K(self.predict_label(x))

    def test(self,
             X_test : np.ndarray,
             T_test : np.ndarray,
             verbose=False) -> float:
        """Test the classifier and return the value of accuracy.
        """
        predicted = self.predict_label(X_test)
        true = utils.digit(T_test)
        
        n_correct = np.count_nonzero(predicted == true)    # 正解サンプル数
        n_sample = len(X_test)                             # 全サンプル数
        accuracy = n_correct / n_sample                    # 正解率

        if verbose:
            print(f"Accuracy: {accuracy * 100:.4f} %")
                
        return accuracy


    

class cluster_mixin:
    estimator_type = 'cluster'

    def check_empty_cluster(self):
        if len(np.unique(self.labels)) < self.k:
            raise EmptyCluster("At least 1 cluster is empty.")

    def set_initial(self, delta):
        """Initialize the centroids of clusters so that a respective distance
        between respective ones can be more than a predetermined value.
        """
        n_sample, n_feature = self.X.shape

        idx = np.random.randint(0, n_sample)
        self.centroids = self.X[[idx]]
        
        while len(self.centroids) < self.k:
            idx = np.random.randint(0, n_sample)
            candidate = self.X[idx]
            if np.all(scipy.linalg.norm(self.centroids - candidate, axis=1) > delta):
                self.centroids = np.vstack([self.centroids, candidate])



class regressor_mixin:
    estimator_type = 'regressor'
