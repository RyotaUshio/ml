from . import base
from .nn import mlp_regressor


class linear_regressor(base._estimator_base, base.regressor_mixin):
    pass
