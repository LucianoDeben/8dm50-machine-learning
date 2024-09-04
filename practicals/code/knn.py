import numpy as np
from collections import Counter

def knn(X, y, x, k):
    # Compute the distance between x and all examples in the training set
    distances = np.sqrt(np.sum((X - x) ** 2, axis=1))

    # Sort by distance and return indices of the first k neighbors
    k_indices = np.argsort(distances)[:k]

    # Extract the labels of the k nearest neighbor training samples
    k_nearest_labels = y[k_indices]

    # Return the most common class label
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]
    