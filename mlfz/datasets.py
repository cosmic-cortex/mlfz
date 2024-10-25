import numpy as np
from .functional.numpy import sigmoid


def generate_clusters(n_clusters, n_features, points_per_cluster, spread=1.0):
    clusters = []
    labels = []
    for i in range(n_clusters):
        mean = np.random.rand(n_features) * 10
        cluster = np.random.normal(
            loc=mean, scale=spread, size=(points_per_cluster, n_features)
        )
        clusters.append(cluster)
        labels.append(np.full(points_per_cluster, i))

    return {"X": np.vstack(clusters), "Y": np.hstack(labels)}


def generate_spiral_dataset(n_points, noise=0.5, twist=380):
    random_points = np.sqrt(np.random.rand(n_points)) * twist * 2 * np.pi / 360

    class_1 = np.column_stack(
        (
            -np.cos(random_points) * random_points + np.random.rand(n_points) * noise,
            np.sin(random_points) * random_points + np.random.rand(n_points) * noise,
        )
    )
    class_2 = np.column_stack(
        (
            np.cos(random_points) * random_points + np.random.rand(n_points) * noise,
            -np.sin(random_points) * random_points + np.random.rand(n_points) * noise,
        )
    )

    X = np.vstack((class_1, class_2))
    Y = np.hstack((np.zeros(n_points), np.ones(n_points))).reshape(-1, 1)

    return {"X": X, "Y": Y}


def generate_linear_regression_ds(n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    a = np.random.rand(n_features)
    b = np.random.rand()
    Y = X @ a + b
    return {"X": X, "Y": Y, "a": a, "b": b}


def generate_linear_classification_ds(n_samples, n_features):
    X = (np.random.rand(n_samples, n_features) - 0.5) * 2
    a = np.random.rand(n_features)
    b = np.random.rand()
    Y = 1 * (sigmoid(X @ a + b) > 0.5)
    return {"X": X, "Y": Y, "a": a, "b": b}
