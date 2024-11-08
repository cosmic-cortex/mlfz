import numpy as np
from scipy.stats import entropy
from graphviz import Digraph
from typing import List, Callable
from ..base import Model


def average_label(Y):
    return np.mean(Y)


def most_frequent_label(Y):
    Y_unique, counts = np.unique(Y, return_counts=True)
    return Y_unique[np.argmax(counts)]


def gini_impurity(p):
    """
    Computes the Gini impurity of the probability distribution p.

    Args:
        p: np.ndarray of categorical variables with shape (n_samples, ),
            representing class labels.

    Returns:
        gi: float, the Gini impurity of the label vector.
    """
    return 1 - (p**2).sum()


def leaf_score(Y, score_fn):
    _, counts = np.unique(Y, return_counts=True)
    p = counts / len(Y)
    return score_fn(p)


def leaf_gini_impurity(Y):
    """
    Computes the Gini impurity of a leaf node.

    Args:
        Y: np.ndarray of categorical variables with shape (n_samples, ),
            representing class labels.

    Returns:
        gi: float, the Gini impurity of the label vector.
    """

    return leaf_score(Y, gini_impurity)


def leaf_entropy(Y):
    return leaf_score(Y, entropy)


def mean_squared_error(Y):
    m = np.mean(Y)
    return np.mean((Y - m) ** 2)


def weighted_score(Ys: List, score_fn: Callable):
    n_samples = sum([len(Y) for Y in Ys])
    return sum([(len(Y) / n_samples) * score_fn(Y) for Y in Ys])


class DecisionTree(Model):
    def __init__(
        self,
        leaf_vote=most_frequent_label,
        leaf_score=leaf_gini_impurity,
        max_depth=None,
        min_samples_split=2,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.leaf_vote = leaf_vote
        self.leaf_score = leaf_score

        self.split_feature_idx = 0
        self.threshold = 0
        self.left_child = None
        self.right_child = None
        self.predicted_class = None

    def __repr__(self):
        return f"{self.__class__.__name__}(max_depth={self.max_depth}, min_samples_split={self.min_samples_split})"

    def _should_stop(self, Y: np.ndarray):
        return (
            (len(np.unique(Y)) == 1)
            or (len(Y) < self.min_samples_split)
            or (self.max_depth == 0)
        )

    def _split_idx(self, X: np.ndarray):
        left_idx = X[:, self.split_feature_idx] < self.threshold
        right_idx = ~left_idx
        return left_idx, right_idx

    def _create_child(self):
        return self.__class__(
            max_depth=self.max_depth - 1,
            min_samples_split=self.min_samples_split,
            leaf_score=self.leaf_score,
            leaf_vote=self.leaf_vote,
        )

    def split(self, X: np.ndarray):
        left_idx, right_idx = self._split_idx(X)
        return X[left_idx], X[right_idx]

    @property
    def is_leaf(self):
        return self.predicted_class != None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if self._should_stop(Y):
            self.predicted_class = self.leaf_vote(Y)
            return self

        X_sorted = np.sort(X, axis=0)
        thresholds = (X_sorted[1:] + X_sorted[:-1]) / 2
        scores = np.zeros_like(thresholds)

        for (i, feature_idx), c in np.ndenumerate(thresholds):
            left_idx = X[:, feature_idx] < c
            right_idx = ~left_idx
            split = [Y[left_idx], Y[right_idx]]
            scores[i, feature_idx] = weighted_score(split, self.leaf_score)

        row_idx, self.split_feature_idx = np.unravel_index(
            np.argmin(scores), scores.shape
        )
        self.threshold = thresholds[row_idx, self.split_feature_idx]

        # recursively training a decision tree
        left_idx, right_idx = self._split_idx(X)

        self.left_child = self._create_child().fit(X[left_idx], Y[left_idx])
        self.right_child = self._create_child().fit(X[right_idx], Y[right_idx])

        return self

    def predict(self, X: np.ndarray):
        if self.is_leaf:
            return np.full(X.shape[0], self.predicted_class)

        predictions = np.zeros(X.shape[0], dtype=np.int32)
        left_idx, right_idx = self._split_idx(X)

        predictions[left_idx] = (
            self.left_child.predict(X[left_idx]) if self.left_child else None
        )
        predictions[right_idx] = (
            self.right_child.predict(X[right_idx]) if self.right_child else None
        )

        return predictions

    @property
    def digraph(self):
        def build_graph(tree, graph, node_id):
            if tree.is_leaf:
                label = (
                    f"{tree.predicted_class:.2f}"
                    if isinstance(tree.predicted_class, float)
                    else f"{tree.predicted_class}"
                )
                graph.node(str(node_id), label=label)
                return

            label = f"X[{tree.split_feature_idx}] < {tree.threshold:.2f}"
            graph.node(str(node_id), label=label)

            left_child_id = node_id * 2 + 1
            right_child_id = node_id * 2 + 2

            if tree.left_child:
                graph.edge(str(node_id), str(left_child_id), label="True")
                build_graph(tree.left_child, graph, left_child_id)

            if tree.right_child:
                graph.edge(str(node_id), str(right_child_id), label="False")
                build_graph(tree.right_child, graph, right_child_id)

        graph = Digraph()
        build_graph(self, graph, node_id=0)
        return graph


class ClassificationTree(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, **kwargs):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            leaf_vote=most_frequent_label,
            leaf_score=leaf_gini_impurity,
        )


class RegressionTree(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_score=1, **kwargs):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            leaf_vote=average_label,
            leaf_score=mean_squared_error,
        )
        self.min_score = min_score

    def _should_stop(self, Y: np.ndarray):
        return (
            (len(np.unique(Y)) == 1)
            or (self.leaf_score(Y) < self.min_score)
            or (len(Y) < self.min_samples_split)
            or (self.max_depth == 0)
        )
