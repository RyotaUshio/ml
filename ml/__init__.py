"""ml - Pattern Recognition & Machine Learning Package for Python
==============================================================
"""

from . import base, exceptions, nn, utils, classify, regress, cluster, feature
from .evaluate import evaluate
from .utils import load, save, load_data, imshow, scatter, monitor

__version__ = '0.0.1'

__all__ = ['base', 'exceptions', 'utils', 'nn', 'classify', 'regress', 'cluster', 'feature', 'evaluate', 'load', 'save', 'load_data', 'imshow', 'scatter']
