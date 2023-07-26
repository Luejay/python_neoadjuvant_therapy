
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone
import numpy as np


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x.best_estimator_) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict_proba(self, X):
        predictions_0 = np.column_stack([
            model.predict_proba(X)[:,0] for model in self.models_
        ])

        predictions_1 = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        means_0 = np.mean(predictions_0, axis=1)
        means_1 = np.mean(predictions_1, axis=1)
        return np.column_stack([means_0, means_1])
