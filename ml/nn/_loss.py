import numpy as np
import dataclasses



@dataclasses.dataclass
class loss_func:
    """Loss function of a neural network.
    """
    
    net: 'mlp' = dataclasses.field(init=False, default=None, repr=False)
    
    """損失関数"""    
    def __call__(self, x, t):
        """損失関数の値.

        Parameter(s)
        ------------
        x : 教師データの入力
        t : 教師データの出力
        """
        if x is not None:
            self.net.forward_prop(x)
        return self._call_impl(t)

    def _call_impl(self, t):
        raise NotImplementedError
            
    def error(self, t):
        """出力層の誤差delta = 出力層の内部ポテンシャルuによる微分
        """
        last_layer = self.net[-1]
        delta = last_layer.z - t
        # mean_squareとsigmoid/softmaxの組み合わせならこれで正しい。ほかの組み合わせでもこれでいいのかは未確認(PRMLのpp.211?まだきちんと追ってない)!!!
        if not last_layer.h.is_canonical:
            if isinstance(self.net.loss, mul_cross_entropy):
                delta = np.matmul(delta.reshape((delta.shape[0], 1, delta.shape[1])), last_layer.h.val2deriv(last_layer.z))
                delta = delta.reshape((delta.shape[0], -1))
            else:
                delta *= last_layer.h.val2deriv(last_layer.z)
        return delta


class mean_square(loss_func):
    """Mean-Square Error.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        SSE = 0.5 * np.linalg.norm(self.net[-1].z - t)**2
        return SSE / batch_size

    
class cross_entropy(loss_func):
    """Cross entropy error for binary classification.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        z = self.net[-1].z
        return -(t * np.log(z) + (1.0 - t) * np.log(1 - z)).sum() / batch_size

    
class mul_cross_entropy(cross_entropy):
    """Cross entropy error for multiclass classification.
    """
    def _call_impl(self, t):
        batch_size = len(t)
        return -np.sum((t * np.log(self.net[-1].z + 1e-7))) / batch_size



LOSSES = {
    'mean_square'       : mean_square,
    'cross_entropy'     : cross_entropy,
    'mul_cross_entropy' : mul_cross_entropy,
    None                : lambda: None
}
