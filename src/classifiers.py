import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.classes_ = None
        self.kdes_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            # Prior probability
            self.priors_[c] = len(X_c) / len(X)
            # KDE for class c
            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            kde.fit(X_c)
            self.kdes_[c] = kde
        return self

    def predict_proba(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            # log P(x | w_c)
            log_p_x_given_c = self.kdes_[c].score_samples(X)
            # log P(x | w_c) + log P(w_c)
            log_probs[:, i] = log_p_x_given_c + np.log(self.priors_[c])
            
        # Convert log probabilities to probabilities using numerically stable softmax
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def get_classifiers_and_grids():
    """
    Returns a dictionary of classifiers and their corresponding hyperparameter grids for tuning.
    """
    classifiers = {
        'Gaussian_Bayes': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        'KNN_Bayes': KNeighborsClassifier(),
        'Parzen_Window': ParzenWindowClassifier(),
        'Logistic_Regression': LogisticRegression(max_iter=1000)
    }
    
    grids = {
        'Gaussian_Bayes': {},
        'KNN_Bayes': {
            'n_neighbors': [1, 3, 5, 7, 11, 15],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        },
        'Parzen_Window': {
            'bandwidth': [0.1, 0.5, 1.0, 2.0, 5.0]
        },
        'Logistic_Regression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }
    
    return classifiers, grids
