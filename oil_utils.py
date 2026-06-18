"""Shared transformers for the oil-adulteration models."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: row-wise centring and scaling of a spectrum."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, float)
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
