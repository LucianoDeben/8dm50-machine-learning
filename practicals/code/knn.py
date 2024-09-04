import numpy as np
from scipy.stats import mode

def knn(X_train, y_train, X_test, k):
    predictions = []
    
    # Compute the distances between each test point and all training points
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        k_neighbors_indices = np.argsort(distances)[:k]
        k_neighbors_labels = y_train[k_neighbors_indices]
        most_common_label = mode(k_neighbors_labels).mode[0]
        predictions.append(most_common_label)
    
    return np.array(predictions)

