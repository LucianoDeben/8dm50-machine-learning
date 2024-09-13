import numpy as np
from scipy.stats import mode

def knn(X_train, y_train, X_test, k, type="classification"):
    """k-Nearest Neighbors algorithm.
    
    Parameters
    ----------
    X_train : array_like
        The training data.
    y_train : array_like
        The target values.
    X_test : array_like
        The test data.
    k : int
        The number of neighbors to consider.
    type : str
        The type of problem. Either 'classification' or 'regression'.
        
    Returns
    -------
    array_like
        The predictions for the test data.
        
    Raises
    ------
    ValueError
        If type is not 'classification' or 'regression'.    
    """   
    predictions = []
    
    # Compute the distances between each test point and all training points
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        k_neighbors_indices = np.argsort(distances)[:k]
        k_neighbors_labels = y_train[k_neighbors_indices]
        
        if type == "classification":
            most_common_label = mode(k_neighbors_labels).mode[0]
            predictions.append(most_common_label)
        elif type == "regression":
            average_target = np.mean(k_neighbors_labels)
            predictions.append(average_target)
        else:
            raise ValueError("Invalid type. Expected 'classification' or 'regression'.")
    
    return np.array(predictions)