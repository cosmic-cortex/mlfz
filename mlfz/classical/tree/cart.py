import numpy as np
from graphviz import Digraph
from typing import List, Callable
from mlfz.classical.base import Model


def most_frequent_label(Y):
    Y_unique, counts = np.unique(Y, return_counts=True)
    return Y_unique[np.argmax(counts)]


def average_label(Y):
    return np.mean(Y)


def gini_impurity(Y):
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


def mean_squared_error(Y):
    m = np.mean(Y)
    return np.mean((Y - m) ** 2)


def weighted_score(Ys: List, score_fn: Callable):
    n_samples = sum([len(Y) for Y in Ys])
    return sum([(len(Y) / n_samples) * score_fn(Y) for Y in Ys])


class DecisionTree(Model):
    def __init__(
        self,
        leaf_vote=gini_impurity,
        leaf_score=most_frequent_label,
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

    def _create_child(self):
        return DecisionTree(
            max_depth=self.max_depth - 1,
            min_samples_split=self.min_samples_split,
            leaf_score=self.leaf_score,
            leaf_vote=self.leaf_vote,
        )

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
            scores[i, feature_idx] = weighted_score(split, gini_impurity)

        row_idx, self.split_feature_idx = np.unravel_index(
            np.argmin(scores), scores.shape
        )
        self.threshold = thresholds[row_idx, self.split_feature_idx]

        # recursively training a
        left_idx, right_idx = self._build_idx(X)

        self.left_child = self._create_child().fit(X[left_idx], Y[left_idx])
        self.right_child = self._create_child().fit(X[right_idx], Y[right_idx])

        return self

    def predict(self, X: np.ndarray):
        if self.is_leaf:
            return np.full(X.shape[0], self.predicted_class)

        predictions = np.zeros(X.shape[0], dtype=np.int32)
        left_idx, right_idx = self._build_idx(X)

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
                label = f"{tree.predicted_class}"
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


def ClassificationTree(max_depth=None, min_samples_split=2):
    return DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        leaf_score=gini_impurity,
        leaf_vote=most_frequent_label,
    )
