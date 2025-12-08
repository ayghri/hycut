from typing import Dict, Literal
import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment
from munkres import Munkres


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """

    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def assign_clusters(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_classes: int,
    num_clusters: int = -1,
) -> np.ndarray:
    """
    Assigns clusters based on the true and predicted cluster assignments.

    Parameters:
    - true_labels: numpy.ndarray, true labels/clusters
    - cluster_assignments: numpy.ndarray, predicted cluster assignments
    - num_classes: int, number of true clusters
    - num_clusters: int, number of predicted clusters (default: num of unique clusters)

    Returns:
    - numpy.ndarray, transform clusters to best matching class label
    """
    unique_clusters_pred = np.unique(cluster_assignments)
    if num_clusters < 0:
        num_clusters = unique_clusters_pred.shape[0]

    cluster_to_index = dict(
        zip(unique_clusters_pred, np.arange(unique_clusters_pred.shape[0]))
    )
    offset_assignments = np.array([cluster_to_index[c] for c in cluster_assignments])
    confusion_matrix = np.eye(num_clusters)[offset_assignments].T.dot(
        np.eye(num_classes)[true_labels]
    )

    # rows, cols = linear_sum_assignment(-confusion_matrix.T)
    # cluster_to_label = dict(zip(rows, cols))
    # print(rows, cols, cluster_to_label)
    cluster_to_label = np.argmax(confusion_matrix, axis=1)
    return np.array([cluster_to_label[c] for c in offset_assignments])


def assign_unique_clusters(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_clusters: int,
) -> np.ndarray:

    c_matrix = metrics.confusion_matrix(true_labels, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(c_matrix, n_clusters=num_clusters)
    indices = np.array(Munkres().compute(cost_matrix))  # pyright: ignore
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    # c_matrix = metrics.confusion_matrix(y, y_pred)
    return y_pred


def evaluate_clustering(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_classes: int,
    num_clusters: int = -1,
) -> Dict:
    """
    Evaluate the clustering performance using various metrics.

    Parameters:
    true_labels (numpy.ndarray): The true cluster labels.
    cluster_assignments (numpy.ndarray): The predicted cluster assignments.
    num_classes (int): The true number of clusters.
    num_clusters (int, optional): The predicted number of clusters. Default is -1.

    Returns:
    dict: A dictionary containing the evaluation metrics including accuracy, NMI, and confusion matrix.
    """
    assert true_labels.shape == cluster_assignments.shape
    if num_clusters == num_classes:
        predicted_labels = assign_clusters(
            true_labels, cluster_assignments, num_clusters
        )
    else:
        predicted_labels = assign_clusters(
            true_labels, cluster_assignments, num_classes, num_clusters
        )
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    return {"accuracy": accuracy, "nmi": nmi, "confusion_matrix": confusion_matrix}


def knn_top_k(
    k: int,
    train_X,
    train_y,
    test_X,
    test_y,
    distance: Literal["cosine", "minkowski"],
    temp=0.07,
    chunksize=None,
    eps=1e-8,
):

    if distance == "cosine":
        train_X = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
        test_X = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)

    num_classes = np.unique(train_y)
    n_trains = train_X.shape[0]
    n_tests = test_X.shape[0]
    if chunksize is None:
        chunksize = n_tests
    chunk_size = min(chunksize, n_tests)
    k = min(k, n_trains)

    top1, top5, total = 0.0, 0.0, 0
    retrieval_one_hot = np.zeros(k, num_classes)
    """

    for idx in range(0, n_tests, chunk_size):
        # get the features for test images
        features = test_X[idx : min((idx + chunk_size), n_tests), :]
        targets = test_y[idx : min((idx + chunk_size), n_tests)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        if distance == "cosine":
            similarities = features.dot(train_X.T)
        else:
            similarities = 1 / (cdist(features, train_X) + eps)

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_y.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        if self.distance_fx == "cosine":
            similarities = similarities.clone().div_(self.T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (
            top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    self.reset()

    """
    return top1, top5


def cluster_acc_score(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def nmi_score(y_true, y_pred):
    nmi = metrics.normalized_mutual_info_score(
        y_true, y_pred, average_method="geometric"
    )
    return nmi


def ari_score(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def ratio_cut_score(W, y, num_clusters):
    """
    Computes the ratio cut value for a given graph.

    Parameters:
    W (numpy.ndarray): Adjacency matrix of the graph.
    y (numpy.ndarray): Cluster assignments for each node.
    num_clusters (int): Number of clusters.
        RC =  sum[ (C^T . W(1-C)).diag() / sum(C, 0) ]
    Returns:
    numpy.ndarray: Ratio cut values for each cluster.
    """
    C = np.eye(num_clusters)[y]
    A = W.dot(C)
    return np.sum(np.diag((1 - C).T.dot(A)) / C.sum(0))
