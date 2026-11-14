import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PersistenceClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        probability = (X["discharge"] >= X["threshold"]).to_numpy(dtype=float)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
