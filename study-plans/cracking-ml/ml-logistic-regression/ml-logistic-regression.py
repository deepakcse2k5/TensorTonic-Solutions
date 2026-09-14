import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    def sigmoid(z):
        return 1/ (1+np.exp(-z))

    for _ in range(n_iters):
        z = X @ weights + bias
        y_pred = sigmoid(z)

        error = y_pred - y

        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) *(np.sum(error))
        weights -= lr*dw
        bias -= lr*db
    return weights.tolist(), float(bias)
