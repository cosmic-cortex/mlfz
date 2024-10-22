import numpy as np
from typing import List
from ..base import Model


def _most_frequent_label(Y):
    Y_unique, counts = np.unique(Y, return_counts=True)
    return Y_unique[np.argmax(counts)]


def gini_impurity_leaf(Y):
    """
    Computes the Gini impurity of a leaf node.

    Args:
        Y: np.ndarray of categorical variables with shape (n_samples, ),
            representing class labels.

    Returns:
        gi: float, the Gini impurity of the label vector.
    """

    _, counts = np.unique(Y, return_counts=True)
    freq = counts / len(Y)
    return 1 - (freq**2).sum()


def gini_impurity_split(Ys: List):
    """
    Computes the Gini impurity of a split.

    Args:
        Ys: list of np.ndarrays with categorical variables representing
            class labels.
    """
    n_samples = sum([len(Y) for Y in Ys])
    return sum([(len(Y) / n_samples) * gini_impurity_leaf(Y) for Y in Ys])


class DecisionTree(Model):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.split_feature_idx = 0
        self.threshold = 0
        self.left_child = None
        self.right_child = None
        self.predicted_class = None

    def _should_stop(self, Y: np.ndarray):
        return (
            (len(np.unique(Y)) == 1)
            or (len(Y) < self.min_samples_split)
            or (self.max_depth == 0)
        )

    def _build_idx(self, X: np.ndarray):
        left_idx = X[:, self.split_feature_idx] < self.threshold
        right_idx = ~left_idx
        return left_idx, right_idx

    @property
    def is_leaf(self):
        return self.predicted_class != None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if self._should_stop(Y):
            self.predicted_class = _most_frequent_label(Y)
            return self

        X_sorted = np.sort(X, axis=0)

        thresholds = (X_sorted[1:] + X_sorted[:-1]) / 2
        scores = np.zeros_like(thresholds)

        for (i, feature_idx), c in np.ndenumerate(thresholds):
            left_idx = X[:, feature_idx] < c
            right_idx = ~left_idx
            scores[i, feature_idx] = gini_impurity_split((Y[left_idx], Y[right_idx]))

        row_idx, self.split_feature_idx = np.unravel_index(
            np.argmin(scores), scores.shape
        )
        self.threshold = thresholds[row_idx, self.split_feature_idx]

        # recursively training a
        left_idx, right_idx = self._build_idx(X)

        self.left_child = DecisionTree(
            max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split
        ).fit(X[left_idx], Y[left_idx])

        self.right_child = DecisionTree(
            max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split
        ).fit(X[right_idx], Y[right_idx])

        return self

    def predict(self, X: np.ndarray):
        if self.is_leaf:
            return np.full(X.shape[0], self.predicted_class)

        left_idx, right_idx = self._build_idx(X)

        predictions = np.zeros(X.shape[0], dtype=np.int32)

        try:
            predictions[left_idx] = self.left_child.predict(X[left_idx])
            predictions[right_idx] = self.right_child.predict(X[right_idx])
        except:
            pass

        return predictions
