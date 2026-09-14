import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_feature = X.shape

    weights = np.zeros(n_feature)
    bias = 0.0

    for _ in range(epochs):
        y_pred = X@ weights + bias
        error = y_pred - y
        dw = (2/n_samples) * (X.T @ error)
        db = (2/n_samples)* np.sum(error)

        weights -= lr*dw
        bias -= lr*db
    return weights, bias
    
