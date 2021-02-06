class EmptyCluster(Exception):
    """Raised if there is at least 1 cluster that no pattern belongs to.
    """
    pass


class NoImprovement(Exception):
    """Raised when no progress is being made any more in training an estimator.
    """
    pass
