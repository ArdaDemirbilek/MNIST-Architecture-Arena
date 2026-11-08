import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Count how often each true class is predicted as each class.
    Rows correspond to true classes and columns to predicted classes,
    so correct predictions lie on the diagonal.

    Args:
        y_true: Integer class labels of shape (num_samples,).
        y_pred: Predicted integer class labels of shape (num_samples,).
        num_classes: Number of classes in the dataset.

    Returns:
        Integer matrix of counts with shape (num_classes, num_classes).
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix