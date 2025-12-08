from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from scipy.optimize import linear_sum_assignment


def unsupervised_contingency(true_labels, cluster_labels):
    """
    Compute the contingency matrix between true and cluster labels and sort it to
    maximize the matching
    Args:
        true_labels (List[int]): True labels
        cluster_labels (List[int]): Cluster labels
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Contingency matrix, row indices, column indices
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    unique_true_labels = np.unique(true_labels)
    unique_cluster_labels = np.unique(cluster_labels)

    contingency_matrix = np.zeros((len(unique_true_labels), len(unique_cluster_labels)))

    for i, true_label in enumerate(unique_true_labels):
        for j, cluster_label in enumerate(unique_cluster_labels):
            contingency_matrix[i, j] = np.sum(
                (true_labels == true_label) & (cluster_labels == cluster_label)
            )

    # Apply the Hungarian algorithm to maximize the matching
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    contingency_matrix = contingency_matrix[row_ind, :][:, col_ind]

    return contingency_matrix, row_ind, col_ind


def unsupervised_accuracy(true_labels, cluster_labels):
    """
    Compute the unsupervised accuracy between true and cluster labels
    Args:
        true_labels (List[int]): True labels
        cluster_labels (List[int]): Cluster labels
    Returns:
        float: Unsupervised accuracy
    """
    contingency_matrix, row_ind, col_ind = unsupervised_contingency(
        true_labels, cluster_labels
    )

    # Calculate the accuracy
    optimal_matching_sum = contingency_matrix[row_ind, col_ind].sum()
    return optimal_matching_sum / len(true_labels)


def nmi_score(true_labels, cluster_labels):
    """
    Compute the normalized mutual information between true and cluster labels
    Args:
        true_labels (List[int]): True labels
        cluster_labels (List[int]): Cluster labels
    Returns:
        float: Normalized mutual information
    """
    return normalized_mutual_info_score(true_labels, cluster_labels)
